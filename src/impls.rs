
use chrono::{Datelike};
use crate::*;
use serde::{Serialize, Serializer};
use std::{fmt, marker::{Copy, PhantomData}};

// === Shared Date Implementations ================================================================

// --- Monthly ------------------------------------------------------------------------------------

/// A date with monthly granularity or larger.
///
/// Client code is responsible for parsing strings into `Monthly`s.
#[derive(Clone, Copy, PartialEq)]
pub struct Monthly {
    pub year: isize,
    pub month: usize,
}

pub enum Month {
    January,
    February,
    March,
    April,
    May,
    June,
    July,
    August,
    September,
    October,
    November,
    December,
}

impl Month {
    pub fn to_num(&self) -> isize {
        match self {
            Month::January => 1,
            Month::February => 2,
            Month::March => 3,
            Month::April => 4,
            Month::May => 5,
            Month::June => 6,
            Month::July => 7,
            Month::August => 8,
            Month::September => 9,
            Month::October => 10,
            Month::November => 11,
            Month::December => 12,
        }
    }
}

impl Date for Monthly {
    fn to_scale(&self) -> Scale<Monthly> {
        Scale {
            scale: (self.year * 12 + (self.month - 1) as isize),
            _phantom: PhantomData,
        }
    }

    fn from_scale(scale: Scale<Monthly>) -> Self {
        Monthly {
            year: scale.scale.div_euclid(12),
            month: (scale.scale % 12 + 1) as usize,
        }
    }

    fn parse_from_str(s: &str) -> Result<Self> {
        let fmt = "%Y-%m-%d";
        let nd = chrono::NaiveDate::parse_from_str(s, fmt)
            .map_err(|_| ParseDateError { s: s.to_owned(), fmt: fmt.to_owned() })?;
        Ok(Monthly::ym(nd.year() as isize, nd.month() as usize))
    }

    fn to_string(&self) -> String {
        format!("{}-{:02}-01", self.year, self.month)
    }
}

// Currently only checked for positive inner value.
impl Monthly {

    /// Return the month of a date.
    pub fn month(&self) -> Month {
        match self.month {
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

    /// Create a monthly date from a year and month.
    pub fn ym(year: isize, month: usize) -> Self {
        Monthly {
            year,
            month,
        }
    }
}

impl fmt::Display for Monthly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{:02}-01", self.year, self.month)
    }
}

impl Serialize for Monthly {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{}-{:02}-01", self.year, self.month))
    }
}

impl fmt::Debug for Monthly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Monthly")
         .field("year", &self.year)
         .field("month", &(self.month))
         .finish()
    }
}

// --- Quarterly ----------------------------------------------------------------------------------

// // === Quarterly ===
// 
// trait FromScale<Scale<A>> for Scale<B>
// where
//     A: Date,
//     B: Date,
// {
//     pub fn from_scale(i32) -> i32;
// 
//     pub fn from(scale: Scale<A>) -> Self {
//         Scale {
//             scale: from(scale.inner()),
//             _phantom: PhantomData,
//         }
//     }
// }
// 
// impl FromScale<Scale<Quarterly>> for Scale<Monthly> {

// === Shared Transform Implementations ===========================================================

// --- SingleF32 ----------------------------------------------------------------------------------

// A time series where each date is associated with a single `f32` of data.
#[derive(Copy, Clone, Debug)]
pub struct SingleF32(pub f32);

impl Value for SingleF32 {

    fn from_csv_string(record: StringRecord) -> Result<Self> {
        check_data_len(&record, 1)?;
        let n: f32 = read_field(&record, 0)?;
        Ok(SingleF32(n))
    }

    fn to_csv_string(&self) -> String {
        format!("{}", self.0)
    }
}

impl fmt::Display for SingleF32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SingleF32({})", self.0)
    }
}

// --- DoubleF32 ----------------------------------------------------------------------------------

// A time series where each date is associated with a single `f32` of data.
#[derive(Copy, Clone, Debug)]
pub struct DoubleF32(pub f32, pub f32);

impl Value for DoubleF32 {

    fn from_csv_string(record: StringRecord) -> Result<Self> {
        check_data_len(&record, 2)?;
        let n1: f32 = record.parse(0)?;
        let n2: f32 = record.parse(1)?;
        Ok(DoubleF32(n1, n2))
    }

    fn to_csv_string(&self) -> String {
        format!("{}", self.0)
    }
}

impl fmt::Display for DoubleF32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DoubleF32({}, {})", self.0, self.1)
    }
}
    
// === Transforms =================================================================================

    // let ts = TimeSeries::from_csv


    // let ts = TimeSeries::<MonthlyDate, 1>::from_csv("./tests/test.csv", "%Y-%m-%d") {
    // 
    // let rts = ts.into_regular(None, None).unwrap();

// Zip one-to-one
//
// We should be able to do it by zipping and mapping.

// pub fn zip_one_one<D, N1, N2, N3>(
//     rts: RegularTimeSeries<D, N1>,
//     other: RegularTimeSeries<D, N2>) -> RegularTimeSeries<D, N3>
// where
//     D: Date,
//     const N1: usize,
//     const N2: usize,
//     const N3: usize,
// {
// 
// 
//     let iter1 = rts.iter();
//     let iter2 = rts.iter();
// }

// Need to align the zips.

#[cfg(test)]
pub mod test {

    use crate::{
        DateRange,
        impls::{DoubleF32, Monthly},
        TimeSeries,
    };

    #[test]
    fn creating_daterange_from_monthly_dates_should_work() {
        let _ = DateRange::new(Monthly::ym(2020, 1), Monthly::ym(2021, 1));
    }

    #[test]
    fn date_should_fail_with_error() {
        let csv = "2018-06-001, 1.2";
        if let Err(err) = TimeSeries::<Monthly, DoubleF32>::from_csv_str(csv) {
            assert_eq!(err.to_string(), "ParseDateError(2018-06-001, %Y-%m-%d)");
        } else { assert!(false) }
    }
}
