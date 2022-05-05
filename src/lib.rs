//! The `time-series` crate constrains a data point to an fixed length array of floating point
//! numbers. Each data point is associated with a date. Dates are unusual in that they map to a
//! regular scale, so that monthly dates are always evenly separated.
//!
//! #### Examples
//!
//! ```ignore
//! use std::convert::TryInto;
//! use std::path::Path;
//! use time_series::{DateRange, MonthlyDate, RegularTimeSeries, TimeSeries};
//!
//! // The standard procedure is to create a `TimeSeries` from `csv` data. `
//! // ::<1> defines the data array to be of length 1.
//! let ts = TimeSeries::<1>::from_csv("./tests/test.csv".into()).unwrap();
//!
//! // And then to convert to a regular time-series with ordered data with regular
//! // intervals and no missing points.
//! let rts: RegularTimeSeries::<1> = ts.try_into().unwrap();
//!
//! // When we create an iterator we define a range of dates to iterate over, or
//! // `None, None` for an open range.
//! let range = DateRange::new(None, Some(MonthlyDate::ym(2013,1)));
//! let iter = rts.iter(range);
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

/// `Duration<MonthlyDate>` represents an interval on the `Scale<MonthlyDate>` scale. It wraps an
/// `isize`.
pub struct Duration<D: Date> {
    delta: isize,
    _phantom: PhantomData<D>,
}

// === Scale ======================================================================================

/// For example, `Scale<MonthlyDate>` is a scale with markers at each month.
#[derive(Copy, Clone, Debug)]
pub struct Scale<D: Date> {
    scale: isize,
    _phantom: PhantomData<D>,
}

// impl<D: Date> Scale<D> {
//     from_isize(i: isize) -> Self {
//         Scale {
//             scale: i,
//             _phantom: PhantomData<D>,
//         }
//     }
// }

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

    fn first_position(&self, date: &D) -> Option<usize> {
        let scale = date.to_scale();

        self.0.iter().position(|date_point| date_point.date().to_scale() == scale)
    }


    // Having an inner function allows from_csv() to read either from a file of from a string.
    // The `path_str` argument contains the file name if it exists, for error messages.
    //
    fn from_csv_inner<R: Read>(mut rdr: Reader<R> , date_fmt: &str, opt_path_str: Option<&str>) -> Result<Self> {

        let mut acc: Vec<DatePoint<D, N>> = Vec::new();

        // rdr doesn't impl Debug

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

    // /// Return the duration between the first and second points.
    // pub fn first_duration(&self) -> Result<Duration<D>> {
    //     if self.0.is_empty() { bail!("Time-series is empty.") }
    //     if self.0.len() == 1 { bail!("Time-series has only one point.") }

    //     let first_date = self.0[0].date;
    //     let second_date = self.0[1].date;

    //     let duration: Duration<D> = first_date.duration(second_date.to_scale());
    //     if duration.delta <= 0 {
    //         bail!(format!("Expected positive duration between {} and {}.", first_date, second_date))
    //     };
    //     Ok(duration)
    // }

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
            Some(i) => { Err(anyhow!(format!("Mismatch at line [{}]", i))) },
            None => Ok(()),
        }
    }
}

// The only way to construct a RegularTimeSeries is by try_into() from a TimeSeries, because this
// checks the length and that durations are uniform.  The general idea is to not change the data
// and only change the structure pointing in to data.

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

/// A time-series with regular, contiguous data.
///
/// A `RegularTimeSeries` is guaranteed to have two or more data points.
#[derive(Debug)]
pub struct RegularTimeSeries<D: Date, const N: usize> {
    range:      DateRange<D>,
    ts:         TimeSeries<D, N>,
}

// impl<const N: usize> Serialize for RegularTimeSeries<N> {
//     fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
//     where
//         S: Serializer,
//     {
//         serializer.serialize_newtype_struct("RegularTimeSeries", &self.ts)
//     }
// }
// 
// impl RegularTimeSeries::<1> {
// 
//     /// Consume two `RegularTimeSeries<1>` and return a `RegularTimeSeries<2>` over a tuple of the
//     /// original values. If the duration of the two time-series' are different then panic. If the
//     /// result has less than two data points then fail.
//     pub fn zip_one_one(self, other: RegularTimeSeries<1>) -> Result<RegularTimeSeries<2>> {
//        
//         // Each TimeSeries is a Vec of DatePoints. We can therefore just do the checks and use a
//         // consuming iterator over all the DatePoints.
// 
//         if self.duration() != other.duration() {
//             bail!(
//                 format!("Expected time-series to have same duration but had [{}] and [{}].",
//                     self.duration(),
//                     other.duration(),
//                 )
//             )
//         };
// 
//         // Find first and last dates, then create iterators with this date range and zip.
// 
//         let first_date = self.first_date().max(other.first_date());  
//         let last_date = self.last_date().min(other.last_date());
// 
//         let date_range = DateRange::new(Some(first_date), Some(last_date));
// 
//         let mut v: Vec<DatePoint<2>> = Vec::new();
// 
//         for (dp1, dp2) in self.iter(date_range).zip(other.iter(date_range)) {
// 
//             // let () = dp1;
// 
//             v.push(DatePoint::<2>::new(dp1.date(), [ dp1.value(0), dp2.value(0) ]));
//         }
// 
//         TimeSeries::<2>::new(v).try_into()
//     }
// }
// 
// 
// // Constrain a `RegularTimeSeries` in-place with a new date range.
// impl<const N: usize> RegularTimeSeries<N> {
//     /// Remove `DatePoints` outside `date_range` from `Self`.
//     pub fn mut_range(&mut self, date_range: &DateRange) {
//         let start_date = match date_range.start_date {
//             None => self.ts.0.first().unwrap().date(),
//             Some(start) => start,
//         };
//         let end_date = match date_range.end_date {
//             None => self.ts.0.last().unwrap().date(),
//             Some(end) => end,
//         };
//         self.ts.0.retain(|dp| dp.date >= start_date && dp.date <= end_date)
//     }
// }
// 
// // Return a new `RegularTimeSeries` with a new date range.
// impl<const N: usize> RegularTimeSeries<N> {
//     /// Remove `DatePoints` outside `date_range` from `Self`.
//     pub fn range(&self, date_range: &DateRange) -> RegularTimeSeries<N> {
//         let start_date = match date_range.start_date {
//             None => self.ts.0.first().unwrap().date(),
//             Some(start) => start,
//         };
//         let end_date = match date_range.end_date {
//             None => self.ts.0.last().unwrap().date(),
//             Some(end) => end,
//         };
//         let mut ts = TimeSeries::new(Vec::new());
//         for dp in self.ts.0.iter() {
//             if dp.date() >= start_date && dp.date() <= end_date {
//                 ts.0.push(*dp)
//             };
//         }
//         ts.try_into().unwrap()
//     }
// }
// 
// impl<const N: usize> RegularTimeSeries<N> {
// 
//     /// Return the datapoint for the given date or error if that date is not in `Self`.
//     pub fn datepoint_from_date(&self, date: MonthlyDate) -> Result<DatePoint::<N>> {
// 
//         if date < self.first_date() || date > self.last_date() { 
//             bail!(format!("Date {:?} not in time-series.", date))
//         };
//         let months_delta = date.as_isize() - self.first_date().as_isize();
//         if months_delta % self.duration.0 != 0 {
//             bail!(format!("Date {:?} not in time-series.", date))
//         };
//         let index = (date.as_isize() - self.first_date().as_isize()) / self.duration.0;
// 
//         Ok(self.ts.0[index as usize])
//     }
// 
//     /// Iterate over some of the data points in a `RegularTimeSeries`.
//     pub fn iter(&self, date_range: DateRange) -> RegularTimeSeriesIter<N> {
//         let ts_start_date = self.ts.0[0].date();
// 
//         let start_date = match date_range.start_date {
//             None => ts_start_date,
//             Some(start) => ts_start_date.max(start),
//         };
// 
//         let ts_end_date = *(&(self.ts.0).last().unwrap().date());
// 
//         let end_date = match date_range.end_date { 
//             None => ts_end_date,
//             Some(end) => ts_end_date.min(end),
//         };
// 
//         RegularTimeSeriesIter {
//             start_date,
//             end_date,
//             date_points: &self.ts.0,
//             counter: 0,
//         }
//     }
// 
//     /// Return the duration between points.
//     pub fn duration(&self) -> Duration {
//         self.duration
//     }
// 
//     /// Return the first point.
//     pub fn first(&self) -> Option<DatePoint<N>> {
//         Some(*self.ts.0.first()?)
//     }
// 
//     /// Return the last point.
//     pub fn last(&self) -> Option<DatePoint<N>> {
//         Some(*self.ts.0.last()?)
//     }
// 
//     /// Take the data at index `n`, and use it to construct a monthly
//     /// time-series from a quarterly time-series, using splines.
//     pub fn to_monthly(&self, n: usize) -> RegularTimeSeries<1> {
// 
//         let x = self.ts.0.iter().map(|dp| dp.date().as_isize() as f64).collect::<Vec<f64>>();
//         let y = self.ts.0.iter().map(|dp| dp.value(n) as f64).collect::<Vec<f64>>();
// 
//         let spline = CubicSpline::from_nodes(x, y);
// 
//         let mut v = Vec::new();
//         for i in self.first_date().as_isize()..=self.last_date().as_isize() {
//             let dp = DatePoint::<1>::new(MonthlyDate(i), [spline.eval(i as f64) as f32]);
//             v.push(dp)
//         };
//         TimeSeries::new(v).try_into().unwrap()
//     }
// 
//     /// Transform a `RegularTimeSeries` into quarterly data.
//     pub fn to_quarterly(&self, n: usize) -> RegularTimeSeries<1> {
// 
//         let x = self.ts.0.iter().map(|dp| dp.date().as_isize() as f64).collect::<Vec<f64>>();
//         let y = self.ts.0.iter().map(|dp| dp.value(n) as f64).collect::<Vec<f64>>();
// 
//         let spline = CubicSpline::from_nodes(x, y);
// 
//         let (add_year, month) = match self.first_date().month_ord() {
//             1           => (0, 1),
//             2 | 3 | 4   => (0, 4),
//             5 | 6 | 7   => (0, 7),
//             8 | 9 | 10  => (0, 10),
//             11 | 12     => (1, 1),
//             _           => panic!(),
//         };
// 
//         let mut date = MonthlyDate::ym(self.first_date().year() + add_year, month);
// 
//         let mut v = Vec::new();
//         while date <= self.last_date() {
//             let dp = DatePoint::<1>::new(date, [spline.eval(date.as_isize() as f64) as f32]);
//             v.push(dp);
//             date = MonthlyDate(date.0 + 3);
//         };
//         TimeSeries::new(v).try_into().unwrap()
//     }
// 
//     /// Transform a `RegularTimeSeries` into year-on-year percentage change over the previous year.
//     pub fn to_year_on_year(&self, n: usize) -> Result<RegularTimeSeries<1>> {
// 
//         let mut v = Vec::new();
//         let mut date = self.first_date();
//         while let Ok(dp2) = self.datepoint_from_date(date + Duration::year()) {
//             let dp1 = self.datepoint_from_date(date).unwrap();
//             let yoy = (dp2.value(0) - dp1.value(n)) * 100.0 / dp1.value(n);
//             let dp = DatePoint::<1>::new(date + Duration::year(), [yoy]);
//             v.push(dp);
//             date = date + self.duration;
//         }
//         TimeSeries::new(v).try_into()
//     }
// 
//     /// Return the maximum of all values at index `n`.
//     pub fn max(&self, n: usize) -> f32 {
//         self.ts.max(n)
//     }
// 
//     /// Return the minimum of all values.
//     pub fn min(&self, n: usize) -> f32 {
//         self.ts.min(n)
//     }
// 
//     /// Return the start date.
//     pub fn first_date(&self) -> MonthlyDate {
//         self.ts.0.first().unwrap().date()
//     }
// 
//     /// Return the end date.
//     pub fn last_date(&self) -> MonthlyDate {
//         self.ts.0.last().unwrap().date()
//     }
// }

/// An iterable range of date.
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
    fn check_contiguous_over() {
        assert!(false)
    }
}
