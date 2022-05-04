
use chrono::{Datelike};
use crate::{Date, Scale};
use serde::{Serialize, Serializer};
use std::{
    // cmp::Ordering,
    // convert::{TryFrom, TryInto},
    fmt,
    // fs,
    marker::{Copy, PhantomData},
    // ops::{Add, Sub},
    // path::Path,
};

// === MonthlyDate ================================================================================

/// A date with monthly granularity or larger.
///
/// Client code is responsible for parsing strings into `MonthlyDate`s.
#[derive(Clone, Copy)]
pub struct MonthlyDate {
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
    fn to_num(&self) -> isize {
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

impl Date for MonthlyDate {
    fn to_scale(&self) -> Scale<MonthlyDate> {
        Scale {
            scale: (self.year * 12 + (self.month - 1) as isize),
            _phantom: PhantomData,
        }
    }

    fn from_scale(scale: Scale<MonthlyDate>) -> Self {
        MonthlyDate {
            year: scale.scale.div_euclid(12),
            month: (scale.scale % 12 + 1) as usize,
        }
    }
}

impl From<chrono::NaiveDate> for MonthlyDate {
    fn from(nd: chrono::NaiveDate) -> Self {
        MonthlyDate::ym(nd.year() as isize, nd.month() as usize)
    }
}

impl Into<chrono::NaiveDate> for MonthlyDate {
    fn into(self) -> chrono::NaiveDate {
        chrono::NaiveDate::from_ymd(self.year as i32, self.month as u32, 1)
    }
}

// Currently only checked for positive inner value.
impl MonthlyDate {

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
        MonthlyDate {
            year,
            month,
        }
    }
}

impl fmt::Display for MonthlyDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{:02}-01", self.year, self.month)
    }
}

impl Serialize for MonthlyDate {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{}-{:02}-01", self.year, self.month))
    }
}

impl fmt::Debug for MonthlyDate  {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MonthlyDate")
         .field("year", &self.year)
         .field("month", &(self.month))
         .finish()
    }
}

pub mod test {

use crate::TimeSeries;
use crate::date_impls::MonthlyDate;

#[test]
fn from_csv() {
    let csv_str = "2020-01-01, 1.2";
    let ts = TimeSeries::<MonthlyDate, 1>::from_csv_str(csv_str, "%Y-%m-%d");
    dbg!(ts);
    assert!(false);
}


}
