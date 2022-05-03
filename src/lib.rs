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

// Date implementations. At the moment there is only one - `MonthlyDate`.
pub mod date_impls;

use anyhow::{
    anyhow,
    bail,
    Context,
    Error,
    Result,
};
use csv::Reader;
// use chrono::{NaiveDate, Datelike};
use fallible_iterator::{
    convert,
};
// use peroxide::numerical::spline::CubicSpline;
use serde::{ Serialize }; // Serializer
use std::{
    cmp::Ordering,
    ffi::OsStr,
    fmt::Display,
    fs,
    marker::{Copy, PhantomData},
    ops::{Add, Sub},
    path::Path,
};

//  TODO: This should also implement Serialize
/// A `Date` can best be thought of as a time_scale, with a pointer to one of the marks on the
/// scale. `Date`s implement `From<chrono::NaiveDate>` and `Into<chrono::NaiveDate>` which provides
/// functionality as as `parse_from_str()` among other things.
pub trait Date
where
    Self: Sized,
    Self: From<chrono::NaiveDate>,
    Self: Into<chrono::NaiveDate>,
    Self: Serialize,
    Self: Display,
    Self: Copy,
{

    fn to_scale(&self) -> Scale<Self>;

    fn from_scale(scale: Scale<Self>) -> Self;
    
    fn duration(&self, scale2: Scale<Self>) -> Duration<Self> {
        Duration::<Self> {
            delta: scale2.scale - self.to_scale().scale,
            _phantom: PhantomData, 
        }  
    }

    fn parse_from_str(fmt: &str, s: &str) -> Result<Self> {
        let nd = chrono::NaiveDate::parse_from_str(fmt, s)?;
        Ok(nd.into())
    }
}

// We want to control the conversion of a Date into a string, and the conversion of a string into a
// Date by application code. We do this by having a fmt string as an argument. 

// A `Duration` represents the distance between two marks on the scale.
pub struct Duration<D: Date> {
    delta: isize,
    _phantom: PhantomData<D>,
}

// === Scale ======================================================================================

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
}

/// A time-series with no guarantees of ordering.
#[derive(Debug, Serialize)]
pub struct TimeSeries<D: Date, const N: usize>(Vec<DatePoint<D, N>>);

impl<D: Date, const N: usize> TimeSeries<D, N> {

    /// Construct a `TimeSeries` from a `Vec` of `DatePoints`.
    pub fn new(v: Vec<DatePoint<D, N>>) -> TimeSeries<D, N> {
        TimeSeries(v)
    }

    /// Push a `DatePoint` onto `Self`.
    pub fn push(&mut self, date_point: DatePoint<D, N>) {
        self.0.push(date_point)
    }

    // Create a new time-series from csv data.
    pub fn from_csv<P: AsRef<OsStr> + ?Sized>(path: &P, date_fmt: &str) -> Result<TimeSeries<D, N>> {

        let path_str = path.as_ref().to_str().unwrap_or("unknown");

        let mut acc: Vec<DatePoint<D, N>> = Vec::new();

        let mut rdr = Reader::from_path(path.as_ref())
            .context(format!("Failed to read file."))?;

        for result_record in rdr.records() {

            let record = result_record?;

            if record.len() != N { 
                match record.position() {
                    Some(pos) => bail!("Record length mismatch at line [{}] in file [{}]", pos.line(), path_str),
                    None => bail!("Record length mismatch at unknown position in file [{}]", path_str),
                }
            };

            let date_str = record.get(0).unwrap_or(
                match record.position() {
                    Some(pos) => bail!("Failed to get date at line [{}] in file [{}]", pos.line(), path_str),
                    None => bail!("Failed to get date at unknown position in file [{}]", path_str),
                }
            );

            let date = <D as Date>::parse_from_str(date_fmt, date_str).unwrap_or(
                match record.position() {
                    Some(pos) => bail!("Failed to parse date at line [{}]", pos.line()),
                    None => bail!("Failed to parse date at unknown position."),
                }
            );

            let mut values = [0f32; N];

            for i in 1..record.len() {
                let num_str = record.get(i).unwrap_or(
                    match record.position() {
                        Some(pos) => {
                            bail!(
                                "Failed to get value in column [{}] at line [{}] in file [{}]",
                                i,
                                pos.line(),
                                path_str
                            )
                        },
                        None => {
                            bail!("Failed to get value at unknown position in file [{}]", path_str)
                        },
                    }
                );
                values[i] = num_str.parse().unwrap_or(
                    match record.position() {
                        Some(pos) => {
                            bail!(
                                "Failed to parse value in column [{}] at line [{}] in file [{}]",
                                i,
                                pos.line(),
                                path_str
                            )
                        },
                        None => {
                            bail!("Failed to parse value at unknown position in file [{}]", path_str)
                        },
                    }
                );
            }

            acc.push(DatePoint::<D, N>::new(date, values));
        }

        Ok(TimeSeries::new(acc))
    }

    /// Return the duration between the first and second points.
    pub fn first_duration(&self) -> Result<Duration<D>> {
        if self.0.is_empty() { bail!("Time-series is empty.") }
        if self.0.len() == 1 { bail!("Time-series has only one point.") }

        let first_date = self.0[0].date;
        let second_date = self.0[1].date;

        let duration: Duration<D> = first_date.duration(second_date.to_scale());
        if duration.delta <= 0 {
            bail!(format!("Expected positive duration between {} and {}.", first_date, second_date))
        };
        Ok(duration)
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
}
// 
// // The only way to construct a RegularTimeSeries is by try_into() from a
// // TimeSeries, because this checks sufficient length and consistent duration.
// /// An iterator over a `RegularTimeSeries`.
// pub struct RegularTimeSeriesIter<'a, const N: usize> {
//     start_date: MonthlyDate,
//     end_date: MonthlyDate,
//     date_points: &'a Vec<DatePoint<N>>,
//     counter: usize,
// }
// 
// impl<'a, const N: usize> Iterator for RegularTimeSeriesIter<'a, N> {
//     type Item = DatePoint<N>;
// 
//     fn next(&mut self) -> Option<Self::Item> {
// 
//         // Beyond the end of self.date_points.
//         if self.counter >= self.date_points.len() {
//             None
//         } else {
// 
//             // Counter points into self.date_points and before start date.
//             if self.date_points[self.counter].date() < self.start_date {
//                 self.counter += 1;
//                 self.next()
// 
//             // Counter points into self.date_points but past end date.
//             } else if self.date_points[self.counter].date() > self.end_date {
//                 return None
// 
//             // Counter points into self.date_points and inside range.
//             } else {
//                 self.counter += 1;
//                 return Some(self.date_points[self.counter - 1])
//             }
//         }
//     }
// }
// 
// /// A time-series with regular, contiguous data.
// ///
// /// A `RegularTimeSeries` is guaranteed to have two or more data points.
// #[derive(Debug)]
// pub struct RegularTimeSeries<const N: usize> {
//     duration:   Duration, 
//     ts:         TimeSeries<N>,
// }
// 
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
//     /// Iterate over the data points in a `RegularTimeSeries`.
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
// 
// /// Specifies the time-span of the data.
// #[derive(Clone, Copy, Debug, Serialize)]
// pub struct DateRange {
//     start_date: Option<MonthlyDate>,
//     end_date:   Option<MonthlyDate>,
// }
// 
// impl DateRange {
// 
//     /// Place a filter on the range of dates. `None` means no constraint is applied.
//     /// ```
//     /// let range = DateRange::new(None, Some(MonthlyDate::ym(2013,1)));
//     /// ```
//     pub fn new(start_date: Option<MonthlyDate>, end_date: Option<MonthlyDate>) -> Self {
//         DateRange {
//             start_date: start_date.clone(),
//             end_date: end_date.clone()
//         }
//     }
// 
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
// 
// 
mod test {

    #[test]
    fn division_should_round_down() {
        assert_eq!(7isize.div_euclid(4), 1);
        assert_eq!((-7isize).div_euclid(4), -2);
    }
}
