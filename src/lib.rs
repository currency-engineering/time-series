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
//! let ts = TimeSeries::<Monthly, SingleF32>::from_csv("./tests/test.csv", "%Y-%m-%d").unwrap();
//!
//! // And then to convert to a regular time-series with ordered data over regular
//! // intervals and no missing points.
//! let rts = ts.into_regular(None, Some(Monthly::ym(2013,1))).unwrap();
//! ```
//! #### Working with `RegularTimeSeries`
//!
//! Most of the transformations we want to make on a `RegularTimesSeries` can be done using it as an
//! iterator and mapping from one `DateValue` to another. We need to do a conversion from a
//! `TimeSeries` to a `RegularTimeSeries` at the end to verify that the time-series is still
//! contiguous.
//!
//! ```ignore
//! let single_column: RegularTimeSeries<Monthly, SingleF32> = regular_time_series
//!     .iter()
//!     .map(|datepoint| datepoint.0)
//!     .collect::<TimeSeries<D, N>>()
//!     .into_regular(None, None).unwrap();
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
//! ```ignore
//! assert_eq!(
//!     (Monthly::ym(2020,12).to_scale() + 1).from_scale(),
//!     Monthly::ym(2020,1),
//! )
//! ```

/// Some common [`Date`](trait.Date.html), Transform, and [`Value>`](traitValue.html)
/// implementations.
pub mod impls;

use thiserror::Error;
use csv::{Reader, ReaderBuilder, Trim};
use serde::{Serialize};
use std::{ cmp::{min, max, Ordering}, convert::TryFrom, fmt::{Debug, Display, self} };
use std::{ io::Read, marker::{Copy, PhantomData}, ops::{Add, Sub} };

type Result<T> = std::result::Result<T, TSError>;

// // === Path Handling Helpers ======================================================================
// 
// fn path_to_string(path: &Path, _err_msg: &str) -> Result<String> {
//     Ok(
//         path.to_str()
//             .ok_or(TSError::FilePath("Failed to read path.".to_owned()))?
//             .to_string()
//     )
// }
// 
// fn osstr_to_string(os_str: &OsStr, err_msg: &str) -> Result<String> {
//     Ok(
//         os_str.to_str()
//             .ok_or(TSError::FilePath("Failed to read path.".to_owned()))?
//             .to_string()
//     ) 
// }

// === Error Handling =============================================================================

// // Given an error message, an optional position in csv string or file and an optional file path,
// // return a full error message.
// fn error_msg(msg: &str, record: &StringRecord, opt_path: Option<&str>) -> String {
//     match (record.position(), opt_path) {
//         (Some(pos), Some(path)) => format!("{} at line [{}] in file [{}]", msg, pos.line(), path),
//         (Some(pos), None) => format!("{} at line [{}]", msg, pos.line()),
//         (None, Some(path)) => format!("{} in file [{}]", msg, path),
//         (None, None) => format!("{}", msg),
//     }
// }

#[derive(Debug, Error)]
pub enum TSError {

    /// Error for if dates are in the wrong order.
    #[error("Start date {0} is later than end date {1}")]
    DateOrder(String, String),

    /// Error for when reconstructing time-series from parts. 
    #[error("The range of dates and the length of the data do not agree.")]
    Parts,

    /// File not found.
    #[error("{0}")]
    FilePath(String),

    /// Error for when dates do not have regular intervals.
    #[error("Non-contiguity between lines {0} and {1}")]
    Irregular(usize, usize),
            
    /// Error for if time-series does not have at least one element.
    #[error("TimeSeries must have at least one element.")]
    Empty,

    /// Error for when there is a mismatch in number of csv fields.
    #[error("{0}")]
    Len(String),

    /// Error if parse error and debug data is available.
    #[error("{0}")]
    CSV(String),

    /// Error for if value does not parse.
    #[error("{0}")]
    ParseValue(String),

    /// Error for if data does not parse.
    #[error("{0}")]
    ParseDate(String),
}

/// A helper for client code to build error messages that may contain a position in the csv file
/// and its file path. For example,
/// ```
/// let pos = csv::Position.set_line(4);
/// let path = "data.csv";
///
/// ts_error = ts_error("Failed to read date", Some(pos.line()), Some(path));
/// assert_eq!(ts_error, "Failed to read date at line [4] from [data.csv]");
/// ```
pub fn ts_error(msg: &str, record: Option<&StringRecord>, path: Option<&str>) -> TSError {
    let pos = record.map(|record| record.position()).flatten();

    TSError::CSV(
        match (pos, path) {
            (Some(pos), Some(path)) => format!("{} at line '{}' from '{}'", msg, pos.line(), path),
            (Some(pos), None) => format!("{} at line '{}'", msg, pos.line()),
            (None, Some(path)) => format!("{} from '{}'", msg, path),
            (None, None) => msg.to_owned(),
        }
    )
}

// === Date trait =================================================================================

//  TODO: This should also implement Serialize
/// A `Date` can best be thought of as a time_scale, with a pointer to one of the marks on the
/// scale. `Date`s implement `From<chrono::NaiveDate>` and `Into<chrono::NaiveDate>` which provides
/// functionality such as `parse_from_str()` among other things.
pub trait Date
where
    Self: Serialize + Debug + Display + Copy,
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
    fn inner(&self) -> isize {
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
    fn eq(&self, other: &Self) -> bool { self.scale == other.scale }
}

impl<D: Date> Eq for Scale<D> {}

// === Value ======================================================================================

/// A `Value` can be anything that can be read from a `csv::StringRecord`.
pub trait Value: Copy {

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
        DateValue {date, data}
    }

    /// Return the date of a `DateValue`.
    pub fn date(&self) -> D {
        self.date
    }

    /// Return an array of data without the date.
    pub fn data(&self) -> V {
        self.data
    }
}

// === TimeSeries =================================================================================

/// A time-series with no guarantees of ordering or unique dates, but must have at least one
/// `DateValue`.
#[derive(Debug, Serialize)]
pub struct TimeSeries<D: Date, V: Value>(Vec<DateValue<D, V>>);

impl<D: Date, V: Value> TimeSeries<D, V> {

    fn first_datepoint(&self) -> DateValue<D, V> {
        *self.0.first().unwrap()
    }

    fn last_datepoint(&self) -> DateValue<D, V> {
        *self.0.last().unwrap()
    }

    // What kind of values can be parsed from csv? See csv crate.

    // Having an inner function allows from_csv() to read either from a file of from a string, or
    // to build a customized csv reader. The `opt_path_str` argument contains the file name if it
    // exists, for error messages.
    fn from_csv_inner<R: Read>(
        mut rdr: Reader<R>,
        date_fmt: &str,
        opt_path: Option<&str>) -> Result<Self> 
    {
        let mut acc: Vec<DateValue<D, V>> = Vec::new();

        let mut field_count: &mut Option<usize> = &mut None; 

        // Iterate over lines of csv
        for result_record in rdr.records() {
            let record = StringRecord(
                result_record
                    .map_err(|_| TSError::ParseDate(format!("Failed to read record")))?
            );

            check_field_count(&record, &mut field_count)?;
            let date: D = date_from_record(&record, opt_path)?;
            let values: V = data_from_record(&record)
                .map_err(|_| TSError::ParseValue("Failed to coerce to data type".to_owned()))?;

            acc.push(DateValue::<D, V>::new(date, values));
        }
        if acc.is_empty() { return Err(TSError::Empty) }
        Ok(TimeSeries(acc))
    }

    /// Create a new time-series from csv file. `date_fmt` specification can be found in the
    /// [chrono crate](https://docs.rs/chrono/latest/chrono/format/strftime/index.html#specifiers).
    pub fn from_csv(path: &str, date_fmt: &str) -> Result<Self> {

        let rdr = ReaderBuilder::new()
            .has_headers(false)
            .trim(Trim::All)
            .from_path(path)
            .map_err(|_| TSError::FilePath(format!("Failed to read file at [{}]", &path)))?;

        TimeSeries::<D, V>::from_csv_inner(rdr, date_fmt, Some(&path))
    }

    /// Create a new time-series from csv file. Usually it is sufficient to use the default
    /// `csv::Reader` but if you need to control the csv reader then you can pass in a
    /// configured `csv::ReaderBuilder`.
    pub fn from_csv_with_builder(
        path: &str,
        date_fmt: &str,
        rdr_builder: ReaderBuilder) -> Result<Self>
    {
        let rdr = rdr_builder
            .from_path(path)
            .map_err(|_| TSError::FilePath(format!("Failed to read file at [{}]", &path)))?;

        TimeSeries::<D, V>::from_csv_inner(rdr, date_fmt, Some(&path))
    }

    /// Create a new time-series from a string in csv format.
    pub fn from_csv_str(csv: &str, date_fmt: &str) -> Result<Self> {

        let rdr = ReaderBuilder::new()
            .has_headers(false)
            .trim(Trim::All)
            .from_reader(csv.as_bytes());

        TimeSeries::<D, V>::from_csv_inner(rdr, date_fmt, None)
    }

    /// Create a new time-series from csv file. Usually it is sufficient to use the default
    /// `csv::Reader` but if more control is required over the CSV reader, then a custom
    /// `csv::ReaderBuilder` can be used as an argument.
    pub fn from_csv_str_with_builder(
        csv: &str,
        date_fmt: &str,
        rdr_builder: ReaderBuilder) -> Result<Self> 
    {
        let rdr = rdr_builder.from_reader(csv.as_bytes());
        TimeSeries::<D, V>::from_csv_inner(rdr, date_fmt, None)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    // Convert a `TimeSeries` into a `RegularTimeSeries` which guarantees that there are no gaps in
    // the data.
    pub fn into_regular(
        self, 
        start_date: Option<D>, 
        end_date: Option<D>) -> Result<RegularTimeSeries<D, V>>
    {
        let range = DateRange::new(
            start_date.unwrap_or(self.first_datepoint().date()),
            end_date.unwrap_or(self.last_datepoint().date()),
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
            Some(i) => Err(TSError::Irregular(i, i+1)),
            None => Ok(()),
        }
    }
}

// fn date_from_csv_error(record: &StringRecord, opt_path: Option<&str>) -> TSError {
//     match (record.position(), opt_path) {
//         (Some(pos), Some(path)) => {TSError::DateFromCSV(format!(
//             "Failed to parse date at line [{}] in file [{}]", pos.line(), path,
//         ))},
//         (Some(pos), None) => {TSError::DateFromCSV(format!(
//             "Failed to parse date at line [{}]", pos.line(),
//         ))},
//         (None, Some(path)) => {TSError::DateFromCSV(format!(
//             "Failed to parse date in file [{}]", path,
//         ))},
//         (None, None) => {TSError::DateFromCSV("Failed to parse date".to_owned())},
//     }
// }

fn date_from_record<D: Date>(record: &StringRecord, _opt_path: Option<&str>) -> Result<D> {
    let line = record.position().map(|pos| format!("{}", pos.line())).unwrap_or("?".to_owned());
    let date_str = record.get(0)
        .ok_or(TSError::ParseDate(format!("Empty record on line [{}]", line)))?;
    <D as Date>::parse_from_str(date_str)
}

fn data_from_record<V: Value>(record: &StringRecord) -> Result<V> {
    let data_record: csv::StringRecord = record.iter().skip(1).collect();
    V::from_csv_string(StringRecord(data_record))
}

// Check that the field count does not change while iterating over records else return an error.
fn check_field_count<'a>(
    record: &StringRecord,
    previous_len: &'a mut Option<usize>) -> Result<()>
{
    match previous_len {
        Some(prev) => {match prev != &mut record.len() {
            true => Err(record_err(&record, *prev)),
            false => Ok(()),
        }},
        None => {
            *previous_len = Some(record.len());
            Ok(())
        },
    }
}

// Return an error message.
fn record_err(record: &StringRecord, previous: usize) -> TSError {
    match record.position() {
        Some(pos) => {TSError::Len(format!(
            "Record at line {} has length {} but previous field has length {}.",
            pos.line(),
            previous,
            record.len()
        ))},
        None => {TSError::Len(format!(
            "Record has length {} but previous field has length {}.", previous, record.len()
        ))},
    }
}

// === RegularTimeSeries ==========================================================================

/// A time-series with regular, contiguous data and at least one `DateValue`.
///
/// A `RegularTimeSeries` is guaranteed to have one or more data points.
#[derive(Debug, Serialize)]
pub struct RegularTimeSeries<D: Date, V: Value> {
    range:  DateRange<D>,
    ts:     TimeSeries<D, V>,
}

impl<D: Date, V: Value> RegularTimeSeries<D, V> {
    /// Returns an iterator over `Self`.
    pub fn iter<'a>(&'a self) -> RegularTimeSeriesIter<'a, D, V> {
        RegularTimeSeriesIter {
            inner_iter: self.range.into_iter(),
            date_points: &self.ts.0,
        } 
    }

    /// Returns an iterator that zips up the common dates in two `RegularTimeSeries`.
    pub fn zip_iter<'a, 'b, V2: Value>(
        &'a self,
        other: &'b RegularTimeSeries<D, V2>) -> ZipIter<'a, 'b, D, V, V2> {
        // The offsets are guaranteed to be positive due to the common() fn, and so can be
        // unwrapped.
        let date_range = self.range.common(&other.range);
        let offset1: usize = (self.range.start.inner() - date_range.start.inner())
            .try_into().unwrap();
        let offset2: usize = (other.range.start.inner() - date_range.start.inner())
            .try_into().unwrap();
        ZipIter {
            inner_iter: date_range.into_iter(), 
            offset1,
            offset2,
            date_points1: &self.ts.0,
            date_points2: &other.ts.0,
        }
    }

    /// Breaks `Self` into raw components. This is useful when building a new `RegularTimeSeries`
    /// with a different scale.
    pub fn into_parts<'a>(self) -> (DateRange<D>, Vec<V>) {
    (
        self.range,
        self.ts.0.iter().map(|dp| dp.data()).collect(),
    )}

    /// Build a `RegularTimeSeries` from a range of dates and data.
    pub fn from_parts(range: DateRange<D>, data: Vec<V>) -> Result<Self> {
        if range.duration() + 1 != data.len() as isize { return Err(TSError::Parts) }
        range.into_iter()
            .zip(data.iter())
            .map(|(date, &data)| DateValue::new(date, data))
            .collect::<TimeSeries<D, V>>()
            .into_regular(None, None)
    }
}

impl<D: Date, V: Value> FromIterator<DateValue<D, V>> for TimeSeries<D, V>
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = DateValue<D, V>>
    {
        Self(iter.into_iter().collect())
    }
}

// === RegularTimeSeriesIter ======================================================================

/// An iterator over a `RegularTimeSeries`.
#[derive(Debug)]
pub struct RegularTimeSeriesIter<'a, D: Date, V: Value>
{
    inner_iter: DateRangeIter<D>,
    date_points: &'a Vec<DateValue<D, V>>,
}

impl<'a, D: Date, V: Value> Iterator for RegularTimeSeriesIter<'a, D, V> {
    type Item = DateValue<D, V>;

    fn next(&mut self) -> Option<Self::Item> {
        match (&mut self.inner_iter).enumerate().next() {
            Some(i) => Some(self.date_points[i.0]),
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
            Some(i) => Some((self.date_points1[i.0 + self.offset1], self.date_points2[i.0 + self.offset2])),
            None => None,
        }
    }
}

// === DateRange ==================================================================================

/// An iterable range of dates.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DateRange<D: Date>{
    start: Scale<D>,
    end: Scale<D>
}

impl<D: Date> DateRange<D> {

    pub fn new(start_date: D, end_date: D) -> Result<Self> {
        let start = start_date.to_scale();
        let end = end_date.to_scale();
        if start > end { return Err(TSError::DateOrder(start.to_string(), end.to_string())) }
        Ok(DateRange { start, end })
    }

    pub fn duration(&self) -> isize {
        self.end.inner() - self.start.inner()
    }

    /// Return a new `DateRange` with dates common to the input `DateRange`s.
    pub fn common(&self, other: &DateRange<D>) -> DateRange<D> {
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
#[derive(Debug)]
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
pub struct StringRecord(csv::StringRecord);

impl StringRecord {

    pub fn inner(&self) -> &csv::StringRecord {
        &self.0
    }

    fn get(&self, i: usize) -> Option<&str> {
        self.0.get(i)
    }

    fn iter(&self) -> StringRecordIter {
        StringRecordIter(
            self.0.iter()
        )
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn position(&self) -> Option<&csv::Position> {
        self.0.position()
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

#[cfg(test)]
mod test {
    use chrono::{Datelike, NaiveDate};
    use crate::*;
    use crate::impls::*;
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
        let ts = TimeSeries::<Monthly, SingleF32>::from_csv_str(csv_str, "%Y-%m-%d").unwrap();
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
        let ts = TimeSeries::<Monthly, SingleF32>::from_csv_str(csv_str, "%Y-%m-%d").unwrap();
        let range = DateRange::new(
            Monthly::ym(2020,1),
            Monthly::ym(2020,3),
        ).unwrap();

        if let Err(e) = ts.check_contiguous_over(&range) {
            assert_eq!(e.to_string(), "Non-contiguity between lines 1 and 2")
        } else { assert!(false) }
    }

    #[test]
    fn from_csv_should_fail_when_wrong_length() {
        let csv_str = "2020-01-01, 1.2";

        if let Err(e) = TimeSeries::<Monthly, DoubleF32>::from_csv_str(csv_str, "%Y-%m-%d") {
            assert_eq!(e.to_string(), "Record length mismatch at line 1")
        } else { assert!(false) }
    }

    #[test]
    fn timeseries_should_have_at_least_one_element() {
        let csv_str = "";
        if let Err(e) = TimeSeries::<Monthly, SingleF32>::from_csv_str(csv_str, "%Y-%m-%d") {
            assert_eq!(e.to_string(), "TimeSeries must have at least one element.")
        } else { assert!(false) }
    }

    #[test]
    fn creating_timeseries_from_csv_should_work() {
        let csv_str = "2020-01-01, 1.2";
        let ts = TimeSeries::<Monthly, SingleF32>::from_csv_str(csv_str, "%Y-%m-%d").unwrap();
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
        let csv_str = indoc! {"
            2020-01-01, 1.2, 4.0
            2021-01-01, 1.3, 4.1
        "};
        let ts = TimeSeries::<Monthly, DoubleF32>::from_csv_str(csv_str, "%Y-%m-%d").unwrap();
        assert_eq!(ts.len(), 2);
    }

    // === DateRange tests =======================================================================

    #[test]
    fn common_daterange_should_work() {
        let dr1 = DateRange::new(Monthly::ym(2018, 6), Monthly::ym(2019, 6)).unwrap();
        let dr2 = DateRange::new(Monthly::ym(2018, 1), Monthly::ym(2019, 1)).unwrap();
        assert_eq!(
            dr1.common(&dr2).start,
            Monthly::ym(2018, 6).to_scale(),
        );
    }
}
