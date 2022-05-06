//! The `time-series` crate constrains a data point to an fixed length array of floating point
//! numbers. Each data point is associated with a date. Dates are unusual in that they map to a
//! regular scale, so that monthly dates are always evenly separated.
//!
//! #### Building a `RegularTimeSeries` from data
//!
//! ```no_run
//! use time_series::{RegularTimeSeries, TimeSeries};
//! use time_series::date_impls::Monthly;
//!
//! // The standard procedure is to create a `TimeSeries` from `csv` data. `
//! // ::<1> defines the data array to be of length 1.
//! let ts = TimeSeries::<Monthly, 1>::from_csv("./tests/test.csv", "%Y-%m-%d").unwrap();
//!
//! // And then to convert to a regular time-series with ordered data with regular
//! // intervals and no missing points.
//! let rts = ts.into_regular(None, Some(Monthly::ym(2013,1))).unwrap();
//! ```
//! #### Mapping a `RegularTimeSeries`
//!
//! Most of the transformations we want to make on a `RegularTimesSeries` can be done using it as an
//! iterator and mapping from one `DatePoint` to another.
//!
//! ```no_run
//! let single_column: RegularTimeSeries<Monthly, 1> = regular_time_series
//!     .iter()
//!     .map(|datepoint| datepoint.from_column(0))
//!     .collect();
//! ```
//!
//! #### Changing the time scale.
//!
//! When we want to do transformations that change the date scaling, we can break the
//! `RegularTimeSeries` into columns and then rebuild a new `RegularTimeSeries` from the parts. 
//!
//! #### Dates
//!
//! Dates that implement the `Date` trait map directly to a `Scale` that maps directly back.
//! `Scale` wraps an integer and provides ordering, comparison and arithmetic functionality. So
//! ```no_run
//! assert_eq!(
//!     (Monthly::ym(2020,12).to_scale() + 1).from_scale(),
//!     Monthly::ym(2020,1),
//! )
//! ```

/// Date implementations. At the moment there is only one - `Monthl`.
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

    /// Parse a `Date` from a string, given a format string.
    fn parse_from_str(fmt: &str, s: &str) -> Result<Self> {
        let nd = chrono::NaiveDate::parse_from_str(fmt, s)?;
        Ok(nd.into())
    }
}

// We want to control the conversion of a Date into a string, and the conversion of a string into a
// Date by application code. We do this by having a fmt string as an argument. 

// === Scale ======================================================================================

/// For example, `Scale<Monthly>`. Scale is used to
/// compare two dates, and add or subtract time units.  
#[derive(Copy, Clone, Debug)]
pub struct Scale<D: Date> {
    scale: isize,
    _phantom: PhantomData<D>,
}

impl<D: Date> Scale<D> {
    fn inner(&self) -> isize {
        self.scale
    }
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
    data: [f32; N],
}

impl<D: Date, const N: usize> DatePoint<D, N> {
    /// Create a new datepoint.
    pub fn new(date: D, data: [f32; N]) -> DatePoint<D, N> {
        DatePoint {date, data}
    }

    /// Return the value at column index.
    pub fn from_column(&self, column: usize) -> DatePoint<D, 1> {
        DatePoint {
            date: self.date,
            data: [self.data[column]]
        }
    }

    /// Return the date of a `DatePoint`.
    pub fn date(&self) -> D {
        self.date
    }

    /// Return an array of data without the date.
    pub fn data(&self) -> [f32; N] {
        self.data
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

// === RegularTimeSeriesIter ======================================================================

/// An iterator over a `RegularTimeSeries`.
pub struct RegularTimeSeriesIter<'a, D: Date, const N: usize> {
    inner_iter: DateRangeIter<D>,
    date_points: &'a Vec<DatePoint<D, N>>,
}

impl<'a, D: Date, const N: usize> Iterator for RegularTimeSeriesIter<'a, D, N> {
    type Item = DatePoint<D, N>;

    fn next(&mut self) -> Option<Self::Item> {
        match (&mut self.inner_iter).enumerate().next() {
            Some(i) => Some(self.date_points[i.0]),
            None => None,
        }
    }
}

// === RegularTimeSeries ==========================================================================

/// A time-series with regular, contiguous data.
///
/// A `RegularTimeSeries` is guaranteed to have one or more data points.
pub struct RegularTimeSeries<D: Date, const N: usize> {
    range:  DateRange<D>,
    ts:     TimeSeries<D, N>,
}

impl<D: Date, const N: usize> RegularTimeSeries<D, N> {

    // // Construct a time-series from a `DateRange` and a `Vec<f32>`.
    // fn from_column(range: DateRange<D>, data: Vec<f32>) -> Result<RegularTimeSeries<D, 1>> {

    //     if range.duration().abs() as usize !=  data.len() - 1 {
    //         bail!("Date range and number of DataPoints do not agree.")
    //     }

    //     let data = range.into_iter()
    //         .zip(data.iter())
    //         .map(|(date, &data)| DatePoint::new(date, [data]))
    //         .collect::<Vec<DatePoint<D, 1>>>();

    //     Ok(RegularTimeSeries { range, ts: TimeSeries::<D, 1>(data) })
    // }

    /// Returns an iterator over `Self`.
    pub fn iter<'a>(&'a self) -> RegularTimeSeriesIter<'a, D, N> {
        RegularTimeSeriesIter {
            inner_iter: self.range.into_iter(),
            date_points: &self.ts.0,
        } 
    }

    /// Breaks `Self` into raw components. This is useful when building a new `RegularTimeSeries`
    /// with a different scale.
    pub fn into_parts<'a>(self) -> (DateRange<D>, Vec<[f32; N]>) {
        (
            self.range,
            self.ts.0.iter().map(|dp| dp.data()).collect(),
        )
    }
}

impl<D: Date, const N: usize> FromIterator<DatePoint<D, N>> for RegularTimeSeries<D, N> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = DatePoint<D, N>>
    {
        let data: Vec<DatePoint<D, N>> = iter.into_iter().collect();

        // This range is always from an iterator, which must be ordered, and so the unwrap()s can
        // be called on it.
        let start_date = data.first().unwrap().date();
        let end_date = data.last().unwrap().date();
        let range = DateRange::new(start_date, end_date).unwrap();

        Self {
            range,
            ts: TimeSeries::<D, N>(data)
        }
    }
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

    pub fn duration(&self) -> isize {
        self.end.inner() - self.start.inner()
    }
}

impl<D: Date> IntoIterator for DateRange<D> {
    type Item = D;
    type IntoIter = DateRangeIter<D>;

    fn into_iter(self) -> Self::IntoIter {
        DateRangeIter {
            count: self.start,
            range: self,
        }
    }
}

/// Iterator over the dates in a range.
pub struct DateRangeIter<D: Date> {
    count: Scale<D>,
    range: DateRange<D>,
}

impl<D: Date> Iterator for DateRangeIter<D> {
    type Item = D;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count <= self.range.end {
            let date = Date::from_scale(self.count);
            self.count = self.count + 1;
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
//     pub fn first_date(&self) -> Option<Monthly> {
//         self.start_date.map(|md| Monthly(md.0))
//     }
// 
//     /// Return the last date. 
//     pub fn last_date(&self) -> Option<Monthly> {
//         self.end_date.map(|md| Monthly(md.0))
//     }
// }
// 
// // pub struct Range {
// //     ts:         &RegularTimeSeries<N>,
// //     start_date: Monthly,
// //     end_date:   Monthly,
// // }
// 

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

#[cfg(test)]
mod test {
    use chrono::{Datelike, NaiveDate};
    use crate::{
        Date,
        date_impls::Monthly,
        DatePoint,
        DateRange,
        RegularTimeSeries,
        Scale,
        TimeSeries,
    };
    use indoc::indoc;

    // === Date trait tests =======================================================================

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

    // === TimeSeries tests =======================================================================

    #[test]
    fn check_contiguous_should_work() {
        let csv_str = indoc! {"
            2020-01-01, 1.2
            2020-02-01, 1.3
            2020-03-01, 1.4
        "};
        let ts = TimeSeries::<Monthly, 1>::from_csv_str(csv_str, "%Y-%m-%d").unwrap();
        let range = DateRange::new(
            Monthly::ym(2020,1),
            Monthly::ym(2020,3),
        ).unwrap();
        if let Ok(()) = ts.check_contiguous_over(&range) { assert!(true) } else { assert!(false) }
    }

    #[test]
    fn check_contiguous_should_fail_correctly() {
        let csv_str = indoc! {"
            2020-01-01, 1.2
            2021-01-01, 1.3
        "};
        let ts = TimeSeries::<Monthly, 1>::from_csv_str(csv_str, "%Y-%m-%d").unwrap();
        let range = DateRange::new(
            Monthly::ym(2020,1),
            Monthly::ym(2020,3),
        ).unwrap();

        if let Err(e) = ts.check_contiguous_over(&range) {
            assert_eq!(e.to_string(), "Non-contiguity between lines [1] and [2]")
        } else { assert!(false) }
    }

    #[test]
    fn from_csv_should_fail_when_wrong_length() {
        let csv_str = "2020-01-01, 1.2";

        if let Err(e) = TimeSeries::<Monthly, 2>::from_csv_str(csv_str, "%Y-%m-%d") {
            assert_eq!(e.to_string(), "Record length mismatch at line [1]")
        } else { assert!(false) }
    }

    #[test]
    fn timeseries_should_have_at_least_one_element() {
        let csv_str = "";
        if let Err(e) = TimeSeries::<Monthly, 1>::from_csv_str(csv_str, "%Y-%m-%d") {
            assert_eq!(e.to_string(), "TimeSeries must have at least one element.")
        } else { assert!(false) }
    }

    #[test]
    fn creating_timeseries_from_csv_should_work() {
        let csv_str = "2020-01-01, 1.2";
        let ts = TimeSeries::<Monthly, 1>::from_csv_str(csv_str, "%Y-%m-%d").unwrap();
        assert_eq!(ts.len(), 1);
    }

    // #[test]
    // fn building_timeseries_from_parts_should_work() {
    //     let date_range = DateRange::new(Monthly::ym(2021, 1), Monthly::ym(2021, 3)).unwrap();
    //     let data = vec!(1.0, 1.1, 1.2); 
    //     let ts = RegularTimeSeries::<Monthly, 1>::from_parts(date_range, data).unwrap();
    //     if let Some(dp) = ts.iter().next() {
    //         assert_eq!(dp.date(), "2020")
    //     }
    // }

    // === DatePoint tests ========================================================================

    #[test]
    fn from_column_works() {
        let dp1 = DatePoint::new(Monthly::ym(2020, 1), [1.2, 4.0]);
        let dp2 = dp1.from_column(0);
        assert_eq!(dp2.date().to_scale(), Monthly::ym(2020,1).to_scale());
        assert_eq!(dp2.data(), [1.2]);
    }
    
    // === RegularTimeSeries tests ================================================================

    #[test]
    fn mapping_a_timeseries_should_work() {
        let csv_str = indoc! {"
            2020-01-01, 1.2, 4.0
            2021-01-01, 1.3, 4.1
        "};
        let ts = TimeSeries::<Monthly, 2>::from_csv_str(csv_str, "%Y-%m-%d").unwrap();
        assert_eq!(ts.len(), 2);
    }
}
