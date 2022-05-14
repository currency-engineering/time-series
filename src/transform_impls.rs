
use time_series::{
    MonthlyDate,
    RegularTimeSeries,
    TimeSeries,
};

pub fn main() {
    // let ts = TimeSeries::from_csv


    // let ts = TimeSeries::<MonthlyDate, 1>::from_csv("./tests/test.csv", "%Y-%m-%d") {
    // 
    // let rts = ts.into_regular(None, None).unwrap();
}

// Zip one-to-one
//
// We should be able to do it by zipping and mapping.

pub fn zip_one_one<D, N1, N2, N3>(
    rts: RegularTimeSeries<D, N1>,
    other: RegularTimeSeries<D, N2>) -> RegularTimeSeries<D, N3>
where
    D: Date,
    const N1: usize,
    const N2: usize,
    const N3: usize,
{


    let iter1 = rts.iter();
    let iter2 = rts.iter();
}

// Need to align the zips.
