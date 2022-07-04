//! The `time-series` crate is an abstraction of dates associated with values. The `Date` trait and
//! the `Value` trait are to be implemented by concrete types. `Date`s are anything that can be
//! mapped to a scale and back again. `Value`s are anything that can be read from a
//! `csv::StringRecord`. Examples of concrete implementations can be found in the `impl` module.
//!
//! #### Building a `RegularTimeSeries` from data
//!
//! ```no_run
//! use time_series::{RegularTimeSeries, TimeSeries};
//! use time_series::impls::{Monthly, SingleF32};
//!
//! // The standard procedure is to create a `TimeSeries` from `csv` data. `
//! let ts = TimeSeries::<Monthly, SingleF32>::from_csv("./tests/test.csv").unwrap();
//!
//! // And then to convert to a regular time-series with ordered data over regular
//! // intervals and no missing points.
//! let rts: RegularTimeSeries::<Monthly, SingleF32> = ts.try_into().unwrap();
//! ```
//! #### Dates
//!
//! Dates that implement the `Date` trait map directly to a `Scale` that maps directly back.
//! `Scale` wraps an integer and provides ordering, comparison and arithmetic functionality. So
//! ```ignore
//! assert_eq!(
//!     (Monthly::ym(2020,12).to_scale() + 1).from_scale(),
//!     Monthly::ym(2020,1),
//! )
//! ```
//!
//! #### Implementations
//!
//! In general the client library will need to implement their own `Value` because each concrete
//! type defines its own format. An example `Value` that reads a single `f32` from a CSV file looks
//! like
//!
//! ```
//! # use std::fmt;
//! # use std::error::Error;
//! # use time_series::{check_data_len, read_field, StringRecord, Value};
//! # type Result<T> = std::result::Result<T, Box<dyn Error + 'static>>;
//!
//! // A time series where each date is associated with a single `f32` of data.
//! #[derive(Copy, Clone, Debug)]
//! pub struct SingleF32(pub f32);
//!
//! impl Value for SingleF32 {
//!
//!     fn from_csv_string(record: StringRecord) -> Result<Self> {
//!         check_data_len(&record, 1)?;
//!         let n: f32 = read_field(&record, 0)?;
//!         Ok(SingleF32(n))
//!     }
//!
//!     fn to_csv_string(&self) -> String {
//!         format!("{}", self.0)
//!     }
//! }
//!
//! impl fmt::Display for SingleF32 {
//!     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//!         write!(f, "SingleF32({})", self.0)
//!     }
//! }
//! ```
//!

pub mod errors;

/// Some common [`Date`](trait.Date.html), Transform, and [`Value>`](traitValue.html)
/// implementations.
pub mod impls;

use crate::errors::*;
use csv::{Reader, ReaderBuilder, Trim};
use serde::Serialize;
use std::error::Error;
use std::{
    cmp::{max, min, Ordering},
    fmt::{self, Debug, Display},
};
use std::{
    io::Read,
    iter::zip,
    marker::{Copy, PhantomData},
    ops::{Add, Sub},
};
use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

type Result<T> = std::result::Result<T, Box<dyn Error + 'static>>;

// === Date trait =================================================================================

//  TODO: This should also implement Serialize
/// A `Date` can best be thought of as a time_scale, with a pointer to one of the marks on the
/// scale. Each CSV format requires its own implementation.
pub trait Date
where
    Self: Serialize + Debug + Display + Copy + PartialEq,
{
    /// Associate a number with every `Date` value.
    fn to_scale(&self) -> Scale<Self>;

    /// Give an `Scale`, return its associated `Date`.
    fn from_scale(scale: Scale<Self>) -> Self;

    fn parse_from_str(s: &str) -> Result<Self>;

    fn to_string(&self) -> String;
}

// === Scale ======================================================================================

/// For example, `Scale<Monthly>`. Scale is used to
/// compare two dates, and add or subtract time units.  
#[derive(Copy, Clone, Debug, Serialize)]
pub struct Scale<D: Date> {
    scale: isize,
    _phantom: PhantomData<D>,
}

impl<D: Date> fmt::Display for Scale<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_date())
    }
}

impl<D: Date> Scale<D> {
    pub fn new(scale: isize) -> Self {
        Scale {
            scale,
            _phantom: PhantomData,
        }
    }

    pub fn inner(&self) -> isize {
        self.scale
    }

    fn to_date(&self) -> D {
        Date::from_scale(*self)
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
    fn eq(&self, other: &Self) -> bool {
        self.scale == other.scale
    }
}

impl<D: Date> Eq for Scale<D> {}

// === Value ======================================================================================

/// A `Value` can be anything that can be read from a `csv::StringRecord`.
pub trait Value: Debug + Display + Copy {
    fn from_csv_string(record: StringRecord) -> Result<Self>
    where
        Self: Sized;

    fn to_csv_string(&self) -> String;
}

// === DateValue ==================================================================================

/// A `Date` associated with a `Value`.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DateValue<D: Date, V: Value> {
    pub date: D,
    data: V,
}

impl<D: Date, V: Value> DateValue<D, V> {
    /// Create a new datepoint.
    pub fn new(date: D, data: V) -> DateValue<D, V> {
        DateValue { date, data }
    }

    /// Return the date of a `DateValue`.
    pub fn date(&self) -> D {
        self.date
    }

    /// Return an array of data without the date.
    pub fn value(&self) -> V {
        self.data
    }
}

// === TimeSeries =================================================================================

/// A time-series with no guarantees of ordering or unique dates, but must have at least one
/// `DateValue`.
#[derive(Debug, Serialize)]
pub struct TimeSeries<D: Date, V: Value>(Vec<DateValue<D, V>>);

impl<D: Date, V: Value> TimeSeries<D, V> {
    // Having an inner function allows from_csv() to read either from a file of from a string, or
    // to build a customized csv reader. The `opt_path_str` argument contains the file name if it
    // exists, for error messages.
    fn from_csv_inner<R: Read>(mut rdr: Reader<R>) -> Result<Self> {
        let mut acc: Vec<DateValue<D, V>> = Vec::new();

        let mut field_count: &mut Option<usize> = &mut None;

        // Iterate over lines of csv
        for res_record in rdr.records() {
            let record = StringRecord(res_record?);

            check_field_count(&record, &mut field_count)?;

            let date: D = date_from_record(&record)?;
            let values: V = data_from_record(&record)?;

            acc.push(DateValue::<D, V>::new(date, values));
        }
        if acc.is_empty() {
            return Err(Box::new(EmptyError()));
        }
        Ok(TimeSeries(acc))
    }

    /// Create a new time-series from csv file. `date_fmt` specification can be found in the
    /// [chrono crate](https://docs.rs/chrono/latest/chrono/format/strftime/index.html#specifiers).
    pub fn from_csv<P: AsRef<Path>>(p: P) -> Result<Self> {
        let path: PathBuf = p.as_ref().to_path_buf();

        let rdr = ReaderBuilder::new()
            .has_headers(false)
            .trim(Trim::All)
            .from_path(path.clone())
            .map_err(|_| Box::new(FilePathError { path }))?;

        TimeSeries::<D, V>::from_csv_inner(rdr)
    }

    /// Create a new time-series from csv file. Usually it is sufficient to use the default
    /// `csv::Reader` but if you need to control the csv reader then you can pass in a
    /// configured `csv::ReaderBuilder`.
    pub fn from_csv_with_builder<P: AsRef<Path>>(p: P, rdr_builder: ReaderBuilder) -> Result<Self> {
        let path = p.as_ref().to_path_buf();
        let rdr = rdr_builder
            .from_path(path.clone())
            .map_err(|_| Box::new(FilePathError { path }))?;

        TimeSeries::<D, V>::from_csv_inner(rdr)
    }

    /// Create a new time-series from a string in csv format.
    pub fn from_csv_str(csv: &str) -> Result<Self> {
        let rdr = ReaderBuilder::new()
            .has_headers(false)
            .trim(Trim::All)
            .from_reader(csv.as_bytes());

        TimeSeries::<D, V>::from_csv_inner(rdr)
    }

    /// Create a new time-series from csv file. Usually it is sufficient to use the default
    /// `csv::Reader` but if more control is required over the CSV reader, then a custom
    /// `csv::ReaderBuilder` can be used as an argument.
    pub fn from_csv_str_with_builder(csv: &str, rdr_builder: ReaderBuilder) -> Result<Self> {
        let rdr = rdr_builder.from_reader(csv.as_bytes());
        TimeSeries::<D, V>::from_csv_inner(rdr)
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    // Fails if dates are not contiguous over the range.
    pub fn check_contiguous(&self) -> Result<()> {
        match self
            .0
            .windows(2)
            .position(|window| window[0].date().to_scale() + 1 != window[1].date().to_scale())
        {
            Some(pos) => Err(Box::new(ContiguousError {
                line_num_opt: Some(pos as u64),
            })),
            None => Ok(()),
        }
    }

    pub fn iter<'a>(&'a self) -> TimeSeriesIter<'a, D, V> {
        TimeSeriesIter {
            data: &self,
            count: 0,
        }
    }
}

impl<D: Date, V: Value> TryFrom<TimeSeries<D, V>> for RegularTimeSeries<D, V> {
    type Error = Box<dyn Error>;

    fn try_from(ts: TimeSeries<D, V>) -> Result<RegularTimeSeries<D, V>> {
        ts.check_contiguous()?;
        let dr = DateRange::new(ts.0.first().unwrap().date(), ts.0.last().unwrap().date())?;
        Ok(RegularTimeSeries {
            range: dr,
            values: ts.iter().map(|dv| dv.value()).collect(),
        })
    }
}

// ===Helper functions=============================================================================

fn date_from_record<D: Date>(record: &StringRecord) -> Result<D> {
    let date_str = record.get(0)?;
    <D as Date>::parse_from_str(&date_str)
}

fn data_from_record<V: Value>(record: &StringRecord) -> Result<V> {
    let data_record: csv::StringRecord = record.iter().skip(1).collect();
    V::from_csv_string(StringRecord(data_record))
}

// Check that the field count does not change while iterating over records else return an error.
fn check_field_count<'a>(record: &StringRecord, previous_len: &'a mut Option<usize>) -> Result<()> {
    match previous_len {
        Some(prev_len) => match prev_len != &mut record.len() {
            true => Err(Box::new(IrregularError {
                line_num_opt: record.line_num_opt(),
                prev_len: *prev_len,
                current_len: record.len(),
            })),
            false => Ok(()),
        },
        None => {
            *previous_len = Some(record.len());
            Ok(())
        }
    }
}

// === TimeSeriesIter ======================================================================

/// An iterator over a `TimeSeries`.
#[derive(Debug)]
pub struct TimeSeriesIter<'a, D: Date, V: Value> {
    data: &'a TimeSeries<D, V>,
    count: usize,
}

impl<'a, D: Date, V: Value> Iterator for TimeSeriesIter<'a, D, V> {
    type Item = DateValue<D, V>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.count < self.data.len() {
            true => {
                self.count += 1;
                Some(self.data.0[self.count - 1])
            }
            false => None,
        }
    }
}

// === RegularTimeSeries ==========================================================================

/// A time-series with regular, contiguous data and at least one `DateValue`.
///
/// A `RegularTimeSeries` is guaranteed to have one or more data points.
#[derive(Debug, Serialize)]
pub struct RegularTimeSeries<D: Date, V: Value> {
    // We do it this way for efficiency
    range: DateRange<D>,
    values: Vec<V>,
}

impl<D: Date, V: Value> RegularTimeSeries<D, V> {
    fn range(&self) -> DateRange<D> {
        self.range
    }

    /// Returns an iterator over `Self`.
    pub fn iter<'a>(&'a self) -> RegularTimeSeriesIter<'a, D, V> {
        RegularTimeSeriesIter {
            inner_iter: self.range.into_iter(),
            values: &self.values,
        }
    }

    // pub fn into_csv(&self) -> String {

    //     fail here
    // }

    // /// Returns an iterator that zips up the common dates in two `RegularTimeSeries`.
    // pub fn zip_iter<'a, 'b, V2: Value>(
    //     &'a self,
    //     other: &'b RegularTimeSeries<D, V2>) -> ZipIter<'a, 'b, D, V, V2> {
    //     // The offsets are guaranteed to be positive due to the common() fn, and so can be
    //     // unwrapped.
    //     let date_range = self.range.common(&other.range);
    //     let offset1: usize = (self.range.start.inner() - date_range.start.inner())
    //         .try_into().unwrap();
    //     let offset2: usize = (other.range.start.inner() - date_range.start.inner())
    //         .try_into().unwrap();
    //     ZipIter {
    //         inner_iter: date_range.into_iter(),
    //         offset1,
    //         offset2,
    //         date_points1: &self.0,
    //         date_points2: &other.0,
    //     }
    // }

    /// Breaks `Self` into raw components. This is useful when building a new `RegularTimeSeries`
    /// with a different scale.
    pub fn into_parts<'a>(self) -> (DateRange<D>, Vec<V>) {
        (self.range(), self.iter().map(|dp| dp.value()).collect())
    }

    /// Build a `RegularTimeSeries` from a range of dates and data.
    pub fn from_parts(range: DateRange<D>, data: Vec<V>) -> Result<Self> {
        if range.duration() + 1 != data.len() as isize {
            return Err(Box::new(PartsError));
        }
        range
            .into_iter()
            .zip(data.iter())
            .map(|(date, &data)| DateValue::new(date, data))
            .collect::<TimeSeries<D, V>>()
            .try_into()
    }
}

impl<D: Date, V: Value> FromIterator<DateValue<D, V>> for TimeSeries<D, V> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = DateValue<D, V>>,
    {
        Self(iter.into_iter().collect())
    }
}

// === RegularTimeSeriesIter ======================================================================

/// An iterator over a `RegularTimeSeries`.
#[derive(Debug)]
pub struct RegularTimeSeriesIter<'a, D: Date, V: Value> {
    inner_iter: DateRangeIter<D>,
    values: &'a Vec<V>,
}

impl<'a, D: Date, V: Value> Iterator for RegularTimeSeriesIter<'a, D, V> {
    type Item = DateValue<D, V>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut iter = zip(self.inner_iter, self.values);
        match iter.next() {
            Some((date, value)) => Some(DateValue::new(date, *value)),
            None => None,
        }
    }
}

// === ZipIter ======================================================================

/// An iterator over the common dates of two `RegularTimeSeries`.
#[derive(Debug)]
pub struct ZipIter<'a, 'b, D: Date, V1: Value, V2: Value> {
    inner_iter: DateRangeIter<D>,
    offset1: usize,
    offset2: usize,
    date_points1: &'a Vec<DateValue<D, V1>>,
    date_points2: &'b Vec<DateValue<D, V2>>,
}

impl<'a, 'b, D: Date, V1: Value, V2: Value> Iterator for ZipIter<'a, 'b, D, V1, V2> {
    type Item = (DateValue<D, V1>, DateValue<D, V2>);

    fn next(&mut self) -> Option<Self::Item> {
        match (&mut self.inner_iter).enumerate().next() {
            Some(i) => Some((
                self.date_points1[i.0 + self.offset1],
                self.date_points2[i.0 + self.offset2],
            )),
            None => None,
        }
    }
}

// === DateRange ==================================================================================

/// An iterable range of dates.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DateRange<D: Date> {
    start: Scale<D>,
    end: Scale<D>,
}

impl<D: Date> DateRange<D> {
    pub fn new(start_date: D, end_date: D) -> Result<Self> {
        let start = start_date.to_scale();
        let end = end_date.to_scale();
        if start > end {
            return Err(Box::new(DateOrderError {
                date1: start.to_string(),
                date2: end.to_string(),
            }));
        }
        Ok(DateRange { start, end })
    }

    pub fn duration(&self) -> isize {
        self.end.inner() - self.start.inner()
    }

    /// Return a new `DateRange` with dates common to the input `DateRange`s.
    pub fn intersection(&self, other: &DateRange<D>) -> DateRange<D> {
        let start = max(self.start, other.start);
        let end = min(self.end, other.end);
        DateRange { start, end }
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

// === DateRangeIter ==============================================================================

/// Iterator over `DateRange`.
#[derive(Copy, Clone, Debug)]
pub struct DateRangeIter<D: Date> {
    count: Scale<D>,
    range: DateRange<D>,
}

impl<D: Date> Iterator for DateRangeIter<D> {
    type Item = D;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count <= self.range.end {
            let date = self.count.to_date();
            self.count = self.count + 1;
            Some(date)
        } else {
            None
        }
    }
}

/// A newtype to wrap `csv::StringRecord`.
#[derive(Debug)]
pub struct StringRecord(csv::StringRecord);

impl StringRecord {
    pub fn as_string(&self) -> String {
        let mut csv_str = String::new();
        for segment in self.0.iter() {
            csv_str.push_str(&segment);
            csv_str.push(',');
        }
        csv_str.pop();
        csv_str
    }

    pub fn inner(&self) -> &csv::StringRecord {
        &self.0
    }

    fn get(&self, i: usize) -> Result<String> {
        match self.0.get(i) {
            Some(s) => Ok(s.to_owned()),
            None => Err(Box::new(FieldError {
                len: self.len(),
                get: i,
            })
            .into()),
        }
    }

    fn parse<T: FromStr>(&self, i: usize) -> Result<T> {
        match self.0.get(i) {
            Some(field) => field.parse().map_err(|_| {
                ParseDataError {
                    line_num_opt: self.line_num_opt(),
                    s: field.to_owned(),
                    get: i,
                }
                .into()
            }),
            None => {
                return Err(Box::new(FieldError {
                    len: self.len(),
                    get: i,
                })
                .into())
            }
        }
    }

    fn iter(&self) -> StringRecordIter {
        StringRecordIter(self.0.iter())
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn position(&self) -> Option<&csv::Position> {
        self.0.position()
    }

    fn line_num_opt(&self) -> Option<u64> {
        self.position().map(|pos| pos.line())
    }
}

impl fmt::Display for StringRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut csv_str = String::new();
        for segment in self.0.iter() {
            csv_str.push_str(&segment);
            csv_str.push(',');
        }
        csv_str.pop();
        write!(f, "{}", csv_str)
    }
}

/// A newtype to wrap `csv::StringRecordIter`.
pub struct StringRecordIter<'r>(csv::StringRecordIter<'r>);

impl<'r> Iterator for StringRecordIter<'r> {
    type Item = &'r str;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

// ===Helper functions=============================================================================

// Check that there are `n` data elements. This does not include the initial date.
pub fn check_data_len(record: &StringRecord, n: usize) -> Result<()> {
    match record.len() != n {
        true => {
            return Err(Box::new(DataLenError {
                line_num_opt: record.line_num_opt(),
                expected_len: n,
                found_len: record.len(),
            }))
        }
        false => Ok(()),
    }
}

pub fn read_field<T: FromStr>(record: &StringRecord, i: usize) -> Result<T> {
    let field = record.inner().get(i).ok_or(Box::new(FieldError {
        len: record.len(),
        get: i,
    }))?;
    field.parse().map_err(|_| {
        Box::new(ParseDataError {
            line_num_opt: record.line_num_opt(),
            s: field.to_owned(),
            get: i,
        })
        .into()
    })
}

// ===Tests========================================================================================

#[cfg(test)]
mod test {
    use crate::impls::*;
    use crate::*;
    use chrono::{Datelike, NaiveDate};

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
        let csv_str = r"
            2020-01-01, 1.2
            2020-02-01, 1.3
            2020-03-01, 1.4";
        let ts = TimeSeries::<Monthly, SingleF32>::from_csv_str(csv_str).unwrap();
        if let Ok(()) = ts.check_contiguous() {
            assert!(true)
        } else {
            assert!(false)
        }
    }

    #[test]
    fn from_csv_should_fail_when_wrong_length() {
        let csv_str = "2020-01-01, 1.2";
        if let Err(err) = TimeSeries::<Monthly, DoubleF32>::from_csv_str(csv_str) {
            assert_eq!(err.to_string(), "DataLenError(None, 2, 1)");
        } else {
            assert!(false)
        }
    }

    #[test]
    fn timeseries_should_have_at_least_one_element() {
        let csv_str = "";
        if let Err(e) = TimeSeries::<Monthly, SingleF32>::from_csv_str(csv_str) {
            assert_eq!(e.to_string(), "EmptyError()")
        } else {
            assert!(false)
        }
    }

    #[test]
    fn creating_timeseries_from_csv_should_work() {
        let csv_str = "2020-01-01, 1.2";
        let ts = TimeSeries::<Monthly, SingleF32>::from_csv_str(csv_str).unwrap();
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

    // === RegularTimeSeries tests ================================================================

    #[test]
    fn mapping_a_timeseries_should_work() {
        let csv_str = "
            2020-01-01, 1.2, 4.0
            2021-01-01, 1.3, 4.1";
        let ts = TimeSeries::<Monthly, DoubleF32>::from_csv_str(csv_str).unwrap();
        assert_eq!(ts.len(), 2);
    }

    // === DateRange tests =======================================================================

    #[test]
    fn intersection_of_dateranges_should_work() {
        let dr1 = DateRange::new(Monthly::ym(2018, 6), Monthly::ym(2019, 6)).unwrap();
        let dr2 = DateRange::new(Monthly::ym(2018, 1), Monthly::ym(2019, 1)).unwrap();
        assert_eq!(
            dr1.intersection(&dr2).start,
            Monthly::ym(2018, 6).to_scale(),
        );
    }

    #[test]
    fn duration_of_daterange_should_work() {
        assert_eq!(
            DateRange::new(Monthly::ym(2018, 6), Monthly::ym(2019, 6))
                .unwrap()
                .duration(),
            12,
        );
        assert_eq!(
            DateRange::new(Monthly::ym(2018, 6), Monthly::ym(2018, 6))
                .unwrap()
                .duration(),
            0,
        );

        if let Err(err) = DateRange::new(Monthly::ym(2018, 6), Monthly::ym(2017, 11)) {
            assert_eq!(err.to_string(), "DateOrderError(2018-06-01, 2017-11-01)");
        } else {
            assert!(false)
        };
    }

    #[test]
    fn duration_should_convert_to_iterator() {
        let mut iter = DateRange::new(Monthly::ym(2018, 6), Monthly::ym(2019, 6))
            .unwrap()
            .into_iter();
        assert_eq!(iter.next(), Some(Monthly::ym(2018, 6)));
        assert_eq!(iter.next(), Some(Monthly::ym(2018, 7)));
    }

    // === StringRecord functions =================================================================

    #[test]
    fn should_get_string_from_stringrecord() {
        let rec = csv::StringRecord::from(vec!["2020-01-01", "1.2", "4.0"]);

        assert_eq!(StringRecord(rec).as_string(), "2020-01-01,1.2,4.0")
    }
}
