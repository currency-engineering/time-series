
use anyhow::{
    anyhow,
    bail,
    Error,
    Result,
};
use chrono::{Datelike};
use crate::{
    StringRecord,
    Date,
    Scale,
};
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

// === Shared Date Implementations ================================================================

// --- Monthly ------------------------------------------------------------------------------------

/// A date with monthly granularity or larger.
///
/// Client code is responsible for parsing strings into `Monthly`s.
#[derive(Clone, Copy)]
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
}

impl From<chrono::NaiveDate> for Monthly {
    fn from(nd: chrono::NaiveDate) -> Self {
        Monthly::ym(nd.year() as isize, nd.month() as usize)
    }
}

impl Into<chrono::NaiveDate> for Monthly {
    fn into(self) -> chrono::NaiveDate {
        chrono::NaiveDate::from_ymd(self.year as i32, self.month as u32, 1)
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

#[cfg(test)]
pub mod test {

    use crate::{
        DateRange,
        date_impls::Monthly,
    };

    #[test]
    fn creating_daterange_from_monthlydates_should_work() {
        DateRange::new(Monthly::ym(2020, 1), Monthly::ym(2021, 1));
    }
}

// === Shared Transform Implementations ===========================================================

pub struct F32(f32);

impl TryFrom<StringRecord> for F32 {
    type Error = Error;

    fn try_from(record: StringRecord) -> Result<Self, Self::Error> {
    
       if record.0.len() != 1 { bail!("Expected records with a date and a single value.") }

       let err_msg = match record.0.position() {
           Some(pos) => format!("Failed to get a singular value on line [{}].", pos.line()),
           None => format!("Failed to get a singular value."),
       };
       
       Ok(F32(
           record.0.get(0)
                .ok_or(anyhow!(err_msg))?
                .parse()?
        ))
    }
}

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
