use std::fmt;

#[macro_export]
macro_rules! err {
    ( $x:expr ) => { err_inner(file!(), line!(), $x) };
}

#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub fn err_inner(file: &str, line: u32, msg: &str) -> Error {
    Error(format!("[{}:{}] {}", file, line, msg))
}


// 
// 
// 
// 
// use std::{
//     error::Error as StdError,
//     fmt,
//     path::PathBuf,
// };
// 
// use crate::{
//     Duration,
//     MonthlyDate
// };
// 
// /// A time_series error.
// #[derive(Debug)]
// pub enum Error {
// 
//     /// The date was not in the time-series.
//     DateNotInTS(MonthlyDate),
// 
//     /// Expected both time-series to have the same durations.
//     NotSameDurations(Duration, Duration),
// 
//     /// Expected positive duration.
//     NonPositiveDuration(MonthlyDate, MonthlyDate),
// 
//     /// Expected regular time-series.
//     NotRegularTS,
// 
//     /// Failed to parse date (path, line_num, line).
//     ParseCSVDate(PathBuf, usize, String),
// 
//     /// Failed to parse csv value.
//     ParseCSVValue(PathBuf, usize, String),
// 
//     /// Failed to read file.
//     ReadFile(PathBuf),
// 
//     /// The time-series has only one point.
//     OnePointTS,
// 
//     // The time-series is empty.
//     EmptyTS,
// }
// 
// impl StdError for Error {}
// 
// impl fmt::Display for Error {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         match self {
//             Error::DateNotInTS(date) => write!(f, "Date {:?} not in time-series.", date),
//             Error::NotSameDurations(duration1, duration2) => {
//                 write!(f,
//                     "Expected time-series to have same duration but had [{}] and [{}].",
//                     duration1,
//                     duration2,
//                 )
//             },
//             Error::NonPositiveDuration(date1, date2) => {
//                 write!(f, "Expected positive duration between {:?} and {:?}.", date1, date2)
//             },
//             Error::NotRegularTS => {
//                 write!(f, "Expected regular time-series.")
//             },
//             Error::ParseCSVDate(path, line_num, line) => {
//                 match path.to_str() {
//                     Some(file) => {
//                         write!(f,
//                             "Line {} file {}. Failed to parse date on line [{}].",
//                             line_num,
//                             file,
//                             line,
//                         )
//                     },
//                     None => {
//                         write!(f,
//                             "Line {}. Failed to parse date on line [{}].",
//                             line_num,
//                             line,
//                         )
//                     },
//                 }
//             },
//             Error::ParseCSVValue(path, line_num, line) => {
//                 match path.to_str() {
//                     Some(file) => {
//                         write!(f,
//                             "Line {} file {}. Failed to parse value on line [{}].",
//                             line_num,
//                             file,
//                             line,
//                         )
//                     },
//                     None => {
//                         write!(f, "Line {}. Failed to parse value on line [{}].",
//                             line_num,
//                             line,
//                         )
//                     },
//                 }
//             },
//             Error::ReadFile(path) => {
//                 match path.to_str() {
//                     Some(file) => write!(f, "Failed to read file [{}].", file),
//                     None => write!(f, "Failed to read file."),
//                 }
//             },
//             Error::OnePointTS => {
//                 write!(f, "Time-series has only one point.")
//             },
//             Error::EmptyTS => {
//                 write!(f, "Time-series is empty.")
//             }
//         }
//     }
// }
// 
// 
