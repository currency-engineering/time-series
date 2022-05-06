//! The `time-series` crate constrains a data point to an fixed length array of floating point
//! numbers. Each data point is associated with a date. Dates are unusual in that they map to a
//! regular scale, so that monthly dates are always evenly separated. Iterator methods can be used
//! to do transformations on time-series.
//!
//! #### Examples
//!
//! ```ignore
//! use time_series::{MonthlyDate, RegularTimeSeries, TimeSeries};
//!
//! // The standard procedure is to create a `TimeSeries` from `csv` data. `
//! // ::<1> defines the data array to be of length 1.
//! let ts = TimeSeries::<MonthlyDate, 1>::from_csv("./tests/test.csv", "%Y-%m-%d") {
//!
//! // And then to convert to a regular time-series with ordered data with regular
//! // intervals and no missing points.
//! let rts = ts.into_regular(None, Some(MonthlyDate::rm(2013,1)).unwrap();
//! ```

/// Date implementations. At the moment there is only one - `MonthlyDate`.
pub mod date_impls;

use anyhow::{
    anyhow,
    bail,
    Context,
    Result,
};
use csv::{Position, Reader, ReaderBuilder, Trim};
// use peroxide::numerical::spline::CubicSpline;
use serde::{ Serialize }; // Serializer
use std::{
    cmp::Ordering,
    ffi::OsStr,
    fmt::{Debug, Display},
    io::Read,
    marker::{Copy, PhantomData},
    ops::{Add, Sub},
};

// === Error Handling =============================================================================

// Given an error message, an optional position in csv string or file and an optional file path,
// return a full error message.
fn csv_error_msg(msg: &str, position: Option<&Position>, opt_path_str: Option<&str>) -> String {
    match (position, opt_path_str) {
        (Some(pos), Some(path)) => format!("{} at line [{}] in file [{}]", msg, pos.line(), path),
        (Some(pos), None) => format!("{} at line [{}]", msg, pos.line()),
        (None, Some(path)) => format!("{} in file [{}]", msg, path),
        (None, None) => format!("{}", msg),
    }
}

// ================================================================================================

//  TODO: This should also implement Serialize
/// A `Date` can best be thought of as a time_scale, with a pointer to one of the marks on the
/// scale. `Date`s implement `From<chrono::NaiveDate>` and `Into<chrono::NaiveDate>` which provides
/// functionality as as `parse_from_str()` among other things.
pub trait Date
where
    Self: Sized + From<chrono::NaiveDate> + Into<chrono::NaiveDate> + Serialize + Debug + Display + Copy,
{

    /// Associate a number with every `Date` value.
    fn to_scale(&self) -> Scale<Self>;

    /// Give an `Scale`, return its associated `Date`.
    fn from_scale(scale: Scale<Self>) -> Self;

    // Do we need this?
    //
    // /// Return the duration between two markers on a scale. The value can be position, zero, or
    // /// negiative.
    // fn duration(&self, scale2: Scale<Self>) -> Duration<Self> {
    //     Duration::<Self> {
    //         delta: scale2.scale - self.to_scale().scale,
    //         _phantom: PhantomData, 
    //     }  
    // }

    /// Parse a `Date` from a string, given a format string.
    fn parse_from_str(fmt: &str, s: &str) -> Result<Self> {
        let nd = chrono::NaiveDate::parse_from_str(fmt, s)?;
        Ok(nd.into())
    }

    /// The name of the unit (in singular) such as "month". Used to format the date.
    fn unit_name() -> &'static str; 
}

// We want to control the conversion of a Date into a string, and the conversion of a string into a
// Date by application code. We do this by having a fmt string as an argument. 

/// `Duration<MonthlyDate>` represents an interval on the `Scale<MonthlyDate>` scale.
pub struct Duration<D: Date> {
    delta: isize,
    _phantom: PhantomData<D>,
}

// === Scale ======================================================================================

/// For example, `Scale<MonthlyDate>` is a scale with markers at each month. Scale is used to
/// compare two dates, and add or subtract time units.  
#[derive(Copy, Clone, Debug)]
pub struct Scale<D: Date> {
    scale: isize,
    _phantom: PhantomData<D>,
}

impl<D: Date> Add<isize> for Scale<D> {
    type Output = Self;

    fn add(self, rhs: isize) -> Self::Output {
        Scale {
            scale: self.scale + rhs,
            _phantom: PhantomData,
        }
    }
}

impl<D: Date> Sub<isize> for Scale<D> {
    type Output = Self;

    fn sub(self, rhs: isize) -> Self::Output {
        Scale {
            scale: self.scale - rhs,
            _phantom: PhantomData,
        }
    }
}

impl<D: Date> PartialOrd for Scale<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.scale.cmp(&other.scale))
    }
}

impl<D: Date> Ord for Scale<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.scale.cmp(&other.scale)
    }
}

impl<D: Date> PartialEq for Scale<D> {
    fn eq(&self, other: &Self) -> bool { self.scale == other.scale }
}

impl<D: Date> Eq for Scale<D> {}

// === DatePoint ==================================================================================

/// A `Date` associated with a fixed length array of `f32`s.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DatePoint<D: Date, const N: usize> {
    pub date: D,
    #[serde(with = "arrays")]
    value: [f32; N],
}

impl<D: Date, const N: usize> DatePoint<D, N> {
    /// Create a new datepoint.
    pub fn new(date: D, value: [f32; N]) -> DatePoint<D, N> {
        DatePoint {date, value}
    }

    /// Return the value at column index.
    pub fn value(&self, column: usize) -> f32 {
        self.value[column]
    }

    /// Return the date of a `DatePoint`.
    pub fn date(&self) -> D {
        self.date
    }
}

// === TimeSeries =================================================================================

/// A time-series with no guarantees of ordering or unique dates, but must have at least one
/// element. The canonical method to create a time-series is from a csv file using
/// `TimeSeries::from_csv("/path/to/data.csv", "%Y-%m-%d")`.
#[derive(Debug, Serialize)]
pub struct TimeSeries<D: Date, const N: usize>(Vec<DatePoint<D, N>>);

impl<D: Date, const N: usize> TimeSeries<D, N> {

    fn first(&self) -> DatePoint<D, N> {
        *self.0.first().unwrap()
    }

    fn last(&self) -> DatePoint<D, N> {
        *self.0.last().unwrap()
    }

    // Having an inner function allows from_csv() to read either from a file of from a string.
    // The `path_str` argument contains the file name if it exists, for error messages.
    //
    fn from_csv_inner<R: Read>(
        mut rdr: Reader<R>,
        date_fmt: &str,
        opt_path_str: Option<&str>) -> Result<Self> 
    {
        let mut acc: Vec<DatePoint<D, N>> = Vec::new();

        // Iterate over lines of csv
        for result_record in rdr.records() {

            // Verify record lengths
            let record = result_record?;
            if record.len() != N + 1 { 
                bail!(csv_error_msg("Record length mismatch", record.position(), opt_path_str))
            }

            // Parse date
            let date_str = record.get(0).context(
                csv_error_msg("Failed to get date", record.position(), opt_path_str)
            )?;

            let date = <D as Date>::parse_from_str(date_str, date_fmt).context(
                csv_error_msg("Failed to parse date", record.position(), opt_path_str)
            )?;

            // Parse values
            let mut values = [0f32; N];
            for i in 1..record.len() {
                let num_str = record.get(i).context(
                    csv_error_msg(
                        &format!("Failed to get value in column [{}]", i),
                        record.position(),
                        opt_path_str
                    )
                )?;
                values[i - 1] = num_str.parse().context(
                    csv_error_msg(
                        &format!("Failed to parse value in column [{}]", i),
                        record.position(),
                        opt_path_str,
                    )
                )?
            }
            acc.push(DatePoint::<D, N>::new(date, values));
        }
        if acc.is_empty() {
            bail!("TimeSeries must have at least one element.")
        }
        Ok(TimeSeries(acc))
    }

    /// Create a new time-series from csv file. `date_fmt` specification can be found in the
    /// [chrono crate](https://docs.rs/chrono/latest/chrono/format/strftime/index.html#specifiers).
    pub fn from_csv<P: AsRef<OsStr> + ?Sized>(path: &P, date_fmt: &str) -> Result<Self> {
        let opt_path_str = path.as_ref().to_str();
        let error_message = match opt_path_str {
            Some(path) => format!("Failed to read file at [{}]", path),
            None => format!("Failed to read file"),
        };

        let rdr = ReaderBuilder::new()
            .has_headers(false)
            .trim(Trim::All)
            .from_path(path.as_ref())
            .context(error_message)?;

        TimeSeries::<D, N>::from_csv_inner(rdr, date_fmt, opt_path_str)
    }

    /// Create a new time-series from csv file. Usually it is sufficient to use the default
    /// `csv::Reader` but if you need to control the csv reader then you can pass in a
    /// configured `csv::ReaderBuilder`.
    pub fn from_csv_with_builder<P: AsRef<OsStr> + ?Sized>(
        path: &P,
        date_fmt: &str,
        rdr_builder: ReaderBuilder) -> Result<Self>
    {
        let opt_path_str = path.as_ref().to_str();
        let error_message = match opt_path_str {
            Some(path) => format!("Failed to read file at [{}]", path),
            None => format!("Failed to read file"),
        };

        let rdr = rdr_builder
            .from_path(path.as_ref())
            .context(error_message)?;

        TimeSeries::<D, N>::from_csv_inner(rdr, date_fmt, opt_path_str)
    }

    /// Create a new time-series from a string in csv format.
    pub fn from_csv_str(csv: &str, date_fmt: &str) -> Result<Self> {

        let rdr = ReaderBuilder::new()
            .has_headers(false)
            .trim(Trim::All)
            .from_reader(csv.as_bytes());

        TimeSeries::<D, N>::from_csv_inner(rdr, date_fmt, None)
    }

    /// Create a new time-series from csv file. Usually it is sufficient to use the default
    /// `csv::Reader` but if you need to control the csv reader then you can pass in a
    /// configured `csv::ReaderBuilder`.
    pub fn from_csv_str_with_builder(csv: &str, date_fmt: &str, rdr_builder: ReaderBuilder) -> Result<Self> {
        let rdr = rdr_builder.from_reader(csv.as_bytes());
        TimeSeries::<D, N>::from_csv_inner(rdr, date_fmt, None)
    }

    // /// Return the maximum of all values at index `n`.
    // pub fn max(&self, n: usize) -> f32 {
    //     self.0.iter()
    //         .map(|dp| dp.value(n))
    //         .fold(f32::NEG_INFINITY, |a, b| a.max(b))
    // }

    // /// Return the minimum of all values at index `n`.
    // pub fn min(&self, n: usize) -> f32 {
    //     self.0.iter()
    //         .map(|dp| dp.value(n))
    //         .fold(f32::INFINITY, |a, b| a.min(b))
    // }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    // Convert a `TimeSeries` into a `RegularTimeSeries` which guarantees that there are no gaps in
    // the data.
    pub fn into_regular(
        self, 
        start_date: Option<D>, 
        end_date: Option<D>) -> Result<RegularTimeSeries<D, N>>
    {
        let range = DateRange::new(
            start_date.unwrap_or(self.first().date()),
            end_date.unwrap_or(self.last().date()),
        )?;

        self.check_contiguous_over(&range)?;
        Ok(RegularTimeSeries { range, ts: self })
    }

    // Fails if dates are not contiguous over the range.
    pub fn check_contiguous_over(&self, range: &DateRange<D>) -> Result<()> {

        // Build time_series iter that removes everything before first date
        let ts_iter = self.0.iter().skip_while(|dp| dp.date().to_scale() != range.start);

        // Zip to date iter and check that dates are the same.
        let check: Option<usize> = range.into_iter()
            .zip(ts_iter)
            .position(|(date, dp)| date.to_scale() != dp.date().to_scale());

        match check {
            Some(i) => { Err(anyhow!(format!("Non-contiguity between lines [{}] and [{}]", i, i+1 ))) },
            None => Ok(()),
        }
    }
}

// ================================================================================================

/// An iterator over a `RegularTimeSeries`.
pub struct RegularTimeSeriesIter<'a, D: Date, const N: usize> {
    start_date: D,
    end_date: D,
    date_points: &'a Vec<DatePoint<D, N>>,
    counter: usize,
}

impl<'a, D: Date, const N: usize> Iterator for RegularTimeSeriesIter<'a, D, N> {
    type Item = DatePoint<D, N>;

    fn next(&mut self) -> Option<Self::Item> {

        // Beyond the end of self.date_points.
        if self.counter >= self.date_points.len() {
            None
        } else {

            // Counter points into self.date_points and before start date.
            if self.date_points[self.counter].date.to_scale() < self.start_date.to_scale() {
                self.counter += 1;
                self.next()

            // Counter points into self.date_points but past end date.
            } else if self.date_points[self.counter].date.to_scale() > self.end_date.to_scale() {
                return None

            // Counter points into self.date_points and inside range.
            } else {
                self.counter += 1;
                return Some(self.date_points[self.counter - 1])
            }
        }
    }
}

// ================================================================================================

/// A time-series with regular, contiguous data.
///
/// A `RegularTimeSeries` is guaranteed to have one or more data points.
pub struct RegularTimeSeries<D: Date, const N: usize> {
    range:      DateRange<D>,
    ts:         TimeSeries<D, N>,
}

// ================================================================================================

/// An iterable range of dates.
#[derive(Clone, Copy, Debug)]
pub struct DateRange<D: Date>{
    start: Scale<D>,
    end: Scale<D>
}

impl<D: Date> DateRange<D> {

    pub fn new(start_date: D, end_date: D) -> Result<Self> {
        let start = start_date.to_scale();
        let end = end_date.to_scale();
        if start > end {
            bail!("Start date [{}] is later than end date [{}]", start_date, end_date)
        }
        Ok(DateRange { start, end })
    }
}

impl<D: Date> IntoIterator for DateRange<D> {
    type Item = D;
    type IntoIter = DateRangeIter<D>;

    fn into_iter(self) -> Self::IntoIter {
        DateRangeIter {
            ptr: self.start,
            range: self,
        }
    }
}

/// Iterator over the dates in a range.
pub struct DateRangeIter<D: Date> {
    ptr: Scale<D>,
    range: DateRange<D>,
}

impl<D: Date> Iterator for DateRangeIter<D> {
    type Item = D;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr <= self.range.end {
            let date = Date::from_scale(self.ptr);
            self.ptr = self.ptr + 1;
            Some(date)
        } else {
            None
        }
    }
}

// pub impl Transform
// where
//     Self: D1: Date,
//     Self: N1: D2: Date, N1: Const, N2: Const> Transform {
// 
//     pub fn from<RegularTransform<D1, N1> for RegularTransform<D2, N2>;
// 
// }

// ================================================================================================

//     /// Return the first date.
//     pub fn first_date(&self) -> Option<MonthlyDate> {
//         self.start_date.map(|md| MonthlyDate(md.0))
//     }
// 
//     /// Return the last date. 
//     pub fn last_date(&self) -> Option<MonthlyDate> {
//         self.end_date.map(|md| MonthlyDate(md.0))
//     }
// }
// 
// // pub struct Range {
// //     ts:         &RegularTimeSeries<N>,
// //     start_date: MonthlyDate,
// //     end_date:   MonthlyDate,
// // }
// 

// This shouldn't be used.
// impl<const N: usize> TryFrom<TimeSeries<N>> for RegularTimeSeries<N> {
//     type Error = Error;
// 
//     fn try_from(ts: TimeSeries<N>) -> std::result::Result<Self, Self::Error> {
// 
//         // This will fail if self has less that 2 datapoints.
//         let duration = ts.first_duration()?;
//         Ok(RegularTimeSeries::<N> { duration, ts })
//     }
// }

// https://github.com/serde-rs/serde/issues/1937

mod arrays {
    use std::{convert::TryInto, marker::PhantomData};

    use serde::{
        de::{SeqAccess, Visitor},
        ser::SerializeTuple,
        Deserialize, Serialize, Serializer,
        // Deserializer
    };
    pub fn serialize<S: Serializer, T: Serialize, const N: usize>(
        data: &[T; N],
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        let mut s = ser.serialize_tuple(N)?;
        for item in data {
            s.serialize_element(item)?;
        }
        s.end()
    }

    struct ArrayVisitor<T, const N: usize>(PhantomData<T>);

    impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
    where
        T: Deserialize<'de>,
    {
        type Value = [T; N];

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str(&format!("an array of length {}", N))
        }

        #[inline]
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            // can be optimized using MaybeUninit
            let mut data = Vec::with_capacity(N);
            for _ in 0..N {
                match (seq.next_element())? {
                    Some(val) => data.push(val),
                    None => return Err(serde::de::Error::invalid_length(N, &self)),
                }
            }
            match data.try_into() {
                Ok(arr) => Ok(arr),
                Err(_) => unreachable!(),
            }
        }
    }

    // pub fn deserialize<'de, D, T, const N: usize>(deserializer: D) -> Result<[T; N], D::Error>
    // where
    //     D: Deserializer<'de>,
    //     T: Deserialize<'de>,
    // {
    //     deserializer.deserialize_tuple(N, ArrayVisitor::<T, N>(PhantomData))
    // }
}
// 
// mod tests {
// 
//     // Private
//     
//     // Duration::year
//     // Duration::new
//     // Duration::is_not_positive
//     // Duration::display
//     
//     // 
// 
//     // MonthlyDate::year
//     // MonthlyDate::month
//     // MonthlyDate::month_ord
//     // MonthlyDate::ym
//     // MonthyDate into_date
//     // MonthlyDate ord
//     // MonthlyDate compare
//     // MonthlyDate eq
//     // MonthlyDate serialize
//     //
//     // Add Duration to MonthlyDate
//     // MonthlyDate debug
//     //
//     // Datepoint::date
//     // Datepoint::new
//     // Datepoint::value
//     //
//     // TimeSeries::new
//     // TimeSeries::push
//     // TimeSeries::from_csv
//     // TimeSeries::first_duration
//     // TimeSeries::max(n)
//     // TimeSeries::min(n)
//     //
//     // RegularTimeSeries from TimeSeries
//     // RegularTimeSeries::serialize
//     // RegularTimeSeries::zip_one_one
//     // RegularTimeSeries::mut_range
//     // RegularTimeSeries::range
//     // RegularTimeSeries::datepoint_from_date
//     // RegularTimeSeries::iter(DateRange)
//     // RegularTimeSeries::duration
//     // RegularTimeSeries::last
//     // RegularTimeSeries::to_monthly
//     // RegularTimeSeries::to_quarterly
//     // RegularTimeSeries::to_year_on_year
//     // RegularTimeSeries::max
//     // RegularTimeSeries::min
//     // RegularTimeSeries::first_date
//     // RegularTimeSeries::last_date
//     //
//     // DateRange
//     //
//     // RegulatTimeSeriesIter
// }
// 
// 
// mod test {
// 
//     #[test]
//     fn duration_year_should_have_12_months() {
//         assert_eq!(Duration::year(), Duration(12));
//     }
//     
//     #[test]
//     fn durations_should_work_across_year_boundaries() {
//         assert_eq!(
//             Duration::new(MonthlyDate::ym(2013, 11), MonthlyDate::ym(2014, 1)),
//             Duration(2)
//         )
//     }
//     
//     #[test]
//     fn durations_can_be_negative() {
//         assert!(
//             Duration::new(
//                 MonthlyDate::ym(2013,4),
//                 MonthlyDate::ym(2013,1),
//             )
//             .is_not_positive()
//         )
//     }
//     
//     #[test]
//     fn duration_should_be_displayable() {
//         assert_eq!(
//             Duration::new(
//                 MonthlyDate::ym(2013,4),
//                 MonthlyDate::ym(2013,1),
//             ).to_string(),
//             "-3 months",
//         )
//     }
// }

#[cfg(test)]
mod test {
    use chrono::{Datelike, NaiveDate};
    use crate::{
        date_impls::MonthlyDate,
        DateRange,
        TimeSeries,
    };
    use indoc::indoc;

    #[test]
    fn division_should_round_down_even_when_numerator_is_negative() {
        assert_eq!(7isize.div_euclid(4), 1);
        assert_eq!((-7isize).div_euclid(4), -2);
    }

    #[test]
    fn date_should_parse() {
        let date = NaiveDate::parse_from_str("2020-01-01", "%Y-%m-%d").unwrap();
        assert!(date.year() == 2020);
    }

    #[test]
    fn check_contiguous_should_work() {
        let csv_str = indoc! {"
            2020-01-01, 1.2
            2020-02-01, 1.3
            2020-03-01, 1.4
        "};
        let ts = TimeSeries::<MonthlyDate, 1>::from_csv_str(csv_str, "%Y-%m-%d").unwrap();
        let range = DateRange::new(
            MonthlyDate::ym(2020,1),
            MonthlyDate::ym(2020,3),
        ).unwrap();
        if let Ok(()) = ts.check_contiguous_over(&range) { assert!(true) } else { assert!(false) }
    }

    #[test]
    fn check_contiguous_should_fail_correctly() {
        let csv_str = indoc! {"
            2020-01-01, 1.2
            2021-01-01, 1.3
        "};
        let ts = TimeSeries::<MonthlyDate, 1>::from_csv_str(csv_str, "%Y-%m-%d").unwrap();
        let range = DateRange::new(
            MonthlyDate::ym(2020,1),
            MonthlyDate::ym(2020,3),
        ).unwrap();

        if let Err(e) = ts.check_contiguous_over(&range) {
            assert_eq!(e.to_string(), "Non-contiguity between lines [1] and [2]")
        } else { assert!(false) }
    }

    #[test]
    fn from_csv_should_fail_when_wrong_length() {
        let csv_str = "2020-01-01, 1.2";

        if let Err(e) = TimeSeries::<MonthlyDate, 2>::from_csv_str(csv_str, "%Y-%m-%d") {
            assert_eq!(e.to_string(), "Record length mismatch at line [1]")
        } else { assert!(false) }
    }

    #[test]
    fn timeseries_should_have_at_least_one_element() {
        let csv_str = "";
        if let Err(e) = TimeSeries::<MonthlyDate, 1>::from_csv_str(csv_str, "%Y-%m-%d") {
            assert_eq!(e.to_string(), "TimeSeries must have at least one element.")
        } else { assert!(false) }
    }

    #[test]
    fn creating_timeseries_from_csv_should_work() {
        let csv_str = "2020-01-01, 1.2";
        let ts = TimeSeries::<MonthlyDate, 1>::from_csv_str(csv_str, "%Y-%m-%d").unwrap();
        assert_eq!(ts.len(), 1);
    }
}
