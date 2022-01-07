//! The `time-series` crate constrains data to an fixed length array of floating point numbers.
//! Each data point is associated with a date. Dates have the special property that they are
//! discrete and separated by an interval such as one month.

pub mod error;

use std::{
    cmp::Ordering,
    convert::{TryFrom, TryInto},
    fmt,
    fs,
    marker::Copy,
    ops::{Add, Sub},
    path::Path,
};
    
use peroxide::numerical::spline::CubicSpline;
use serde::{ Serialize, Serializer };
use time::{ Date, Month };

use crate::error::*;

pub type Result<T> = std::result::Result<T, Error>;

/// A duration between two `Months`.
///
/// The value can be positive or negative.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Duration(isize);

impl Duration {

    fn year(n: isize) -> Self {
        Duration(12 * n)
    }

    /// Return the duration between two dates.
    fn between(min: MonthlyDate, max: MonthlyDate) -> Self {
        Duration(max.as_isize() - min.as_isize())
    }

    fn is_not_positive(&self) -> bool {
        self.0 <= 0
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        match self.0 {
            1 => s.push_str("1 month"),
            n => {
                s.push_str(&n.to_string());
                s.push_str(" months");
            },
        };
        write!(f, "{}", s)
    }
}

/// A date with monthly granularity or larger.
///
/// Client code is responsible for parsing strings into `MonthlyDate`s.
#[derive(Clone, Copy, Eq)]
pub struct MonthlyDate(pub isize);

// Currently only checked for positive inner value.
impl MonthlyDate {
    /// Return the year of a date.
    pub fn year(&self) -> isize {
        self.0 / 12
    }

    /// Return the month of a date.
    pub fn month(&self) -> Month {
        match self.month_ord() {
            1 => Month::January,
            2 => Month::February,
            3 => Month::March,
            4 => Month::April,
            5 => Month::May,
            6 => Month::June,
            7 => Month::July,
            8 => Month::August,
            9 => Month::September,
            10 => Month::October,
            11 => Month::November,
            12 => Month::December,
            _ => panic!(),
        }
    }

    /// Return a monthly ordinal from 1 to 12.
    pub fn month_ord(&self) -> usize {
        (self.0 % 12 + 1) as usize
    }

    /// Return the inner value of a date. This value is an integer representing
    /// the total number of months. 
    pub fn as_isize(&self) -> isize {
        self.0
    }

    /// Create a monthly date from a year and month.
    pub fn ym(year: isize, month: usize) -> Self {
        MonthlyDate(year * 12 + (month - 1) as isize)
    }
}

impl Into<Date> for MonthlyDate {
    fn into(self) -> Date {
        Date::from_calendar_date(
            self.year().try_into().unwrap(),
            self.month(),
            1,
        ).unwrap()
    }
}

impl PartialOrd for MonthlyDate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.cmp(&other.0))
    }
}

impl Ord for MonthlyDate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialEq for MonthlyDate {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl Serialize for MonthlyDate {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{}-{:02}-01", self.year(), self.month()))
    }
}

impl Add<Duration> for MonthlyDate {
    type Output = MonthlyDate;

    fn add(self, duration: Duration) -> Self {
        MonthlyDate(self.as_isize() + duration.0)
    }
}

impl Sub<Duration> for MonthlyDate {
    type Output = MonthlyDate;

    fn sub(self, duration: Duration) -> Self {
        MonthlyDate(self.as_isize() - duration.0)
    }
}

impl fmt::Debug for MonthlyDate  {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MonthlyDate")
         .field("year", &self.year())
         .field("month", &(self.month()))
         .finish()
    }
}

/// A `MonthlyDate` associated with some data.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DatePoint<const N: usize> {
    date: MonthlyDate,
    #[serde(with = "arrays")]
    value: [f32; N],
}

impl<const N: usize> DatePoint<N> {

    /// Return the date of a datepoint.
    pub fn date(&self) -> MonthlyDate { self.date }

    /// Create a new datepoint.
    pub fn new(date: MonthlyDate, value: [f32; N]) -> DatePoint<N> {
        DatePoint {date, value}
    }

    /// Return the value at index `n`.
    pub fn value(&self, n: usize) -> f32 {
        self.value[n]
    }
}

/// A time-series with no guarantees of ordering.
#[derive(Debug, Serialize)]
pub struct TimeSeries<const N: usize>(Vec<DatePoint<N>>);

impl<const N: usize> TimeSeries<N> {

    /// Construct a `TimeSeries` from a `Vec` of `DatePoints`.
    pub fn new(v: Vec<DatePoint<N>>) -> TimeSeries<N> {
        TimeSeries(v)
    }

    /// Push a `DatePoint` onto `Self`.
    pub fn push(&mut self, date_point: DatePoint<N>) {
        self.0.push(date_point)
    }

    /// From CSV file with format '2017-01-01, 4.725'.
    pub fn from_csv(path: &Path) -> Result<TimeSeries<1>> {

        let s = fs::read_to_string(path)
            .map_err(|_| err!(
                &format!("Failed to read file [{}].", path.to_path_buf().to_str().unwrap())
            ))?;

        let mut v: Vec<DatePoint<1>> = Vec::new();
        for (i, line) in s.lines().enumerate() {

            let year = line[..4].parse()
                .map_err(|_| {
                    err!(
                        &format!(
                            "Line {:?} file {}. Failed to parse value on line [{}].",
                            path.to_path_buf(),
                            i + 1,
                            String::from(line)
                        )
                    )
                })?;

            let month = line[5..7].parse()
                .map_err(|_| {
                    err!(
                        &format!(
                            "Line {:?} file {}. Failed to parse value on line [{}].",
                            path.to_path_buf(),
                            i + 1,
                            String::from(line)
                        )
                    )
                })?;

            let value = line[12..].parse::<f32>()
                .map_err(|_| {
                    err!(
                        &format!(
                            "Line {:?} file {}. Failed to parse value on line [{}].",
                            path.to_path_buf(),
                            i + 1,
                            String::from(line)
                        )
                    )
                })?;

            let dp = DatePoint::<1>::new(
                MonthlyDate::ym(year, month),
                [value],
            );

            v.push(dp);
        }

        Ok(TimeSeries::new(v))
    }

    /// Return the duration between the first and second points.
    pub fn first_duration(&self) -> Result<Duration> {

        if self.0.is_empty() { return Err(err!("Time-series is empty.")) }

        if self.0.len() == 1 { return Err(err!("Time-series has only one point.")) }

        let first_date = self.0[0].date();
        let second_date = self.0[1].date();

        let duration = Duration::between(first_date, second_date);
        if duration.is_not_positive() {
            return Err(err!(
                &format!("Expected positive duration between {:?} and {:?}.", first_date, second_date)
            ))
        };
        Ok(duration)
    }

    /// Return the maximum of all values at index `n`.
    pub fn max(&self, n: usize) -> f32 {
        self.0.iter()
            .map(|dp| dp.value(n))
            .fold(f32::NEG_INFINITY, |a, b| a.max(b))
    }

    /// Return the minimum of all values at index `n`.
    pub fn min(&self, n: usize) -> f32 {
        self.0.iter()
            .map(|dp| dp.value(n))
            .fold(f32::INFINITY, |a, b| a.min(b))
    }
}

// The only way to construct a RegularTimeSeries is by try_into() from a
// TimeSeries, because this checks sufficient length and consistent duration.
/// An iterator over a `RegularTimeSeries`.
pub struct RegularTimeSeriesIter<'a, const N: usize> {
    start_date: MonthlyDate,
    end_date: MonthlyDate,
    date_points: &'a Vec<DatePoint<N>>,
    counter: usize,
}

impl<'a, const N: usize> Iterator for RegularTimeSeriesIter<'a, N> {
    type Item = DatePoint<N>;

    fn next(&mut self) -> Option<Self::Item> {

        // Beyond the end of self.date_points.
        if self.counter >= self.date_points.len() {
            None
        } else {

            // Counter points into self.date_points and before start date.
            if self.date_points[self.counter].date() < self.start_date {
                self.counter += 1;
                self.next()

            // Counter points into self.date_points but past end date.
            } else if self.date_points[self.counter].date() > self.end_date {
                return None

            // Counter points into self.date_points and inside range.
            } else {
                self.counter += 1;
                return Some(self.date_points[self.counter - 1])
            }
        }
    }
}

#[test]
fn test_iter() {
    let date1 = MonthlyDate::ym(1995, 11);
    let date2 = MonthlyDate::ym(1995, 12);
    let date3 = MonthlyDate::ym(1996, 1);
    let date4 = MonthlyDate::ym(1996, 2);
    let date5 = MonthlyDate::ym(1996, 3);

    let dp1 = DatePoint::new(date1, [1.2]);
    let dp2 = DatePoint::new(date2, [1.4]);
    let dp3 = DatePoint::new(date3, [1.6]);
    let dp4 = DatePoint::new(date4, [1.8]);
    let dp5 = DatePoint::new(date5, [2.0]);

    let v = vec!( dp1, dp2, dp3, dp4, dp5);

    let rts: RegularTimeSeries<1> = TimeSeries::new(v).try_into().unwrap();
    let date_range = DateRange::new(&Some(date2), &Some(date4));
    let mut iter = rts.iter(date_range);
    assert_eq!(iter.next().unwrap().date(), date2);
    assert_eq!(iter.next().unwrap().date(), date3);
    assert_eq!(iter.next().unwrap().date(), date4);
    assert!(iter.next().is_none());
}

/// A time-series with regular, contiguous data.
///
/// A `RegularTimeSeries` is guaranteed to have two or more data points.
#[derive(Debug)]
pub struct RegularTimeSeries<const N: usize> {
    duration:   Duration, 
    ts:         TimeSeries<N>,
}

impl<const N: usize> Serialize for RegularTimeSeries<N> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_newtype_struct("RegularTimeSeries", &self.ts)
    }
}

impl RegularTimeSeries::<1> {

    /// Consume two `RegularTimeSeries<1>` and return a `RegularTimeSeries<2>` over a tuple of the
    /// original values. If the duration of the two time-series' are different then panic. If the
    /// result has less than two data points then fail.
    pub fn zip_one_one(self, other: RegularTimeSeries<1>) -> Result<RegularTimeSeries<2>> {
       
        // Each TimeSeries is a Vec of DatePoints. We can therefore just do the checks and use a
        // consuming iterator over all the DatePoints.

        if self.duration() != other.duration() {
            return Err(err!(
                &format!("Expected time-series to have same duration but had [{}] and [{}].",
                    self.duration(),
                    other.duration(),
                )
            ))
        };

        // Find first and last dates, then create iterators with this date range and zip.

        let first_date = self.first_date().max(other.first_date());  
        let last_date = self.last_date().min(other.last_date());

        let date_range = DateRange::new(&Some(first_date), &Some(last_date));

        let mut v: Vec<DatePoint<2>> = Vec::new();

        for (dp1, dp2) in self.iter(date_range).zip(other.iter(date_range)) {

            // let () = dp1;

            v.push(DatePoint::<2>::new(dp1.date(), [ dp1.value(0), dp2.value(0) ]));
        }

        TimeSeries::<2>::new(v).try_into()
    }
}


// Constrain a `RegularTimeSeries` in-place with a new date range.
impl<const N: usize> RegularTimeSeries<N> {
    /// Remove `DatePoints` outside `date_range` from `Self`.
    pub fn mut_range(&mut self, date_range: &DateRange) {
        let start_date = match date_range.start_date {
            None => self.ts.0.first().unwrap().date(),
            Some(start) => start,
        };
        let end_date = match date_range.end_date {
            None => self.ts.0.last().unwrap().date(),
            Some(end) => end,
        };
        self.ts.0.retain(|dp| dp.date >= start_date && dp.date <= end_date)
    }
}

// Return a new `RegularTimeSeries` with a new date range.
impl<const N: usize> RegularTimeSeries<N> {
    /// Remove `DatePoints` outside `date_range` from `Self`.
    pub fn range(&self, date_range: &DateRange) -> RegularTimeSeries<N> {
        let start_date = match date_range.start_date {
            None => self.ts.0.first().unwrap().date(),
            Some(start) => start,
        };
        let end_date = match date_range.end_date {
            None => self.ts.0.last().unwrap().date(),
            Some(end) => end,
        };
        let mut ts = TimeSeries::new(Vec::new());
        for dp in self.ts.0.iter() {
            if dp.date() >= start_date && dp.date() <= end_date {
                ts.0.push(*dp)
            };
        }
        ts.try_into().unwrap()
    }
}

#[test]
fn test_with_range() {
    let date1 = MonthlyDate::ym(1995, 11);
    let date2 = MonthlyDate::ym(1995, 12);
    let date3 = MonthlyDate::ym(1996, 1);
    let date4 = MonthlyDate::ym(1996, 2);
    let date5 = MonthlyDate::ym(1996, 3);

    let dp1 = DatePoint::new(date1, [1.2]);
    let dp2 = DatePoint::new(date2, [1.4]);
    let dp3 = DatePoint::new(date3, [1.6]);
    let dp4 = DatePoint::new(date4, [1.8]);
    let dp5 = DatePoint::new(date5, [2.0]);

    let v = vec!( dp1, dp2, dp3, dp4, dp5);

    let mut rts: RegularTimeSeries<1> = TimeSeries::new(v).try_into().unwrap();

    let date_range = DateRange::new(&Some(date2), &Some(date4));

    rts.with_range(&date_range); 

    let mut iter = rts.iter(DateRange::new(&None, &None));
    assert_eq!(iter.next().unwrap().date(), date2);
    assert_eq!(iter.next().unwrap().date(), date3);
    assert_eq!(iter.next().unwrap().date(), date4);
    assert!(iter.next().is_none());
}

impl<const N: usize> RegularTimeSeries<N> {

    /// Return the datapoint for the given date or error if that date is not in `Self`.
    pub fn datepoint_from_date(&self, date: MonthlyDate) -> Result<DatePoint::<N>> {

        if date < self.first_date() || date > self.last_date() { 
            return Err(err!(&format!("Date {:?} not in time-series.", date)))
        };
        let months_delta = date.as_isize() - self.first_date().as_isize();
        if months_delta % self.duration.0 != 0 {
            return Err(err!(&format!("Date {:?} not in time-series.", date)))
        };
        let index = (date.as_isize() - self.first_date().as_isize()) / self.duration.0;

        Ok(self.ts.0[index as usize])
    }

    /// Iterate over the data points in a `RegularTimeSeries`.
    pub fn iter(&self, date_range: DateRange) -> RegularTimeSeriesIter<N> {
        let ts_start_date = self.ts.0[0].date();

        let start_date = match date_range.start_date {
            None => ts_start_date,
            Some(start) => ts_start_date.max(start),
        };

        let ts_end_date = *(&(self.ts.0).last().unwrap().date());

        let end_date = match date_range.end_date { 
            None => ts_end_date,
            Some(end) => ts_end_date.min(end),
        };

        RegularTimeSeriesIter {
            start_date,
            end_date,
            date_points: &self.ts.0,
            counter: 0,
        }
    }

    /// Return the duration between points.
    pub fn duration(&self) -> Duration {
        self.duration
    }

    /// Return the first point.
    pub fn first(&self) -> Option<DatePoint<N>> {
        Some(*self.ts.0.first()?)
    }

    /// Return the last point.
    pub fn last(&self) -> Option<DatePoint<N>> {
        Some(*self.ts.0.last()?)
    }

    /// Take the data at index `n`, and use it to construct a monthly
    /// time-series from a quarterly time-series, using splines.
    pub fn to_monthly(&self, n: usize) -> RegularTimeSeries<1> {

        let x = self.ts.0.iter().map(|dp| dp.date().as_isize() as f64).collect::<Vec<f64>>();
        let y = self.ts.0.iter().map(|dp| dp.value(n) as f64).collect::<Vec<f64>>();

        let spline = CubicSpline::from_nodes(x, y);

        let mut v = Vec::new();
        for i in self.first_date().as_isize()..=self.last_date().as_isize() {
            let dp = DatePoint::<1>::new(MonthlyDate(i), [spline.eval(i as f64) as f32]);
            v.push(dp)
        };
        TimeSeries::new(v).try_into().unwrap()
    }

    /// Transform a `RegularTimeSeries` into quarterly data.
    pub fn to_quarterly(&self, n: usize) -> RegularTimeSeries<1> {

        let x = self.ts.0.iter().map(|dp| dp.date().as_isize() as f64).collect::<Vec<f64>>();
        let y = self.ts.0.iter().map(|dp| dp.value(n) as f64).collect::<Vec<f64>>();

        let spline = CubicSpline::from_nodes(x, y);

        let (add_year, month) = match self.first_date().month_ord() {
            1           => (0, 1),
            2 | 3 | 4   => (0, 4),
            5 | 6 | 7   => (0, 7),
            8 | 9 | 10  => (0, 10),
            11 | 12     => (1, 1),
            _           => panic!(),
        };

        let mut date = MonthlyDate::ym(self.first_date().year() + add_year, month);

        let mut v = Vec::new();
        while date <= self.last_date() {
            let dp = DatePoint::<1>::new(date, [spline.eval(date.as_isize() as f64) as f32]);
            v.push(dp);
            date = MonthlyDate(date.0 + 3);
        };
        TimeSeries::new(v).try_into().unwrap()
    }

    /// Transform a `RegularTimeSeries` into year-on-year percentage change over the previous year.
    pub fn to_year_on_year(&self, n: usize) -> Result<RegularTimeSeries<1>> {

        let mut v = Vec::new();
        let mut date = self.first_date();
        while let Ok(dp2) = self.datepoint_from_date(date + Duration::year(1)) {
            let dp1 = self.datepoint_from_date(date).unwrap();
            let yoy = (dp2.value(0) - dp1.value(n)) * 100.0 / dp1.value(n);
            let dp = DatePoint::<1>::new(date + Duration::year(1), [yoy]);
            v.push(dp);
            date = date + self.duration;
        }
        TimeSeries::new(v).try_into()
    }

    /// Return the maximum of all values at index `n`.
    pub fn max(&self, n: usize) -> f32 {
        self.ts.max(n)
    }

    /// Return the minimum of all values.
    pub fn min(&self, n: usize) -> f32 {
        self.ts.min(n)
    }

    /// Return the start date.
    pub fn first_date(&self) -> MonthlyDate {
        self.ts.0.first().unwrap().date()
    }

    /// Return the end date.
    pub fn last_date(&self) -> MonthlyDate {
        self.ts.0.last().unwrap().date()
    }
}

// Shouldn't we make a external RegularTimeSeriesIter ? It points into RegularTimeSeries and has a flag which
// manages the dates.


/// Specifies the time-span of the data.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DateRange {
    start_date: Option<MonthlyDate>,
    end_date:   Option<MonthlyDate>,
}

impl DateRange {

    /// Return a new `DateRange`.
    pub fn new(start_date: &Option<MonthlyDate>, end_date: &Option<MonthlyDate>) -> Self {
        DateRange {
            start_date: start_date.clone(),
            end_date: end_date.clone()
        }
    }

    /// Return the first date.
    pub fn first_date(&self) -> Option<MonthlyDate> {
        self.start_date.map(|md| MonthlyDate(md.0))
    }

    /// Return the last date. 
    pub fn last_date(&self) -> Option<MonthlyDate> {
        self.end_date.map(|md| MonthlyDate(md.0))
    }
}

// pub struct Range {
//     ts:         &RegularTimeSeries<N>,
//     start_date: MonthlyDate,
//     end_date:   MonthlyDate,
// }

impl<const N: usize> TryFrom<TimeSeries<N>> for RegularTimeSeries<N> {
    type Error = Error;

    fn try_from(ts: TimeSeries<N>) -> std::result::Result<Self, Self::Error> {

        // This will fail if self has less that 2 datapoints.
        let duration = ts.first_duration()?;
        Ok(RegularTimeSeries::<N> { duration, ts })
    }
}

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

