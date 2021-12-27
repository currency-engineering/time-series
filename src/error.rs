use std::fmt;
use std::path::Path;

/// A time_series error.
#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for Error {}

///
pub fn date_not_in_timeseries(
    code_file: &str,
    code_line: u32,
    date: &str,
    ) -> Error
{
    Error(format!(
        "[time_series:01:{}:{}] Date {} not in time-series.",
        code_file,
        code_line,
        date
    ))
}

pub fn expected_same_durations(
    code_file: &str,
    code_line: u32,
    first_duration: &str,
    second_duration: &str) -> Error
{
    Error(format!(
        "[time_series:02:{}:{}] Expected time-series to have same duration but had [{}] and [{}].",
        code_file,
        code_line,
        first_duration,
        second_duration,
    ))
}

pub fn expected_positive_duration(
    code_file: &str,
    code_line: u32,
    date1: &str,
    date2: &str) -> Error
{
    Error(format!(
        "[time_series:02:{}:{}] Expected positive duration between {} and {}.",
        code_file,
        code_line,
        date1,
        date2
    ))
}

///
pub fn expected_regular_time_series() -> Error {
    Error(format!(
        "[time_series:03] Expected regular time-series."
    ))
}

///
pub fn parse_csv_date_failed(
    code_file: &str,
    code_line: u32,
    file: &Path,
    line_num: usize,
    line: &str) -> Error
{
    match file.to_str() {
        
        // If possible to parse Path to str.
        Some(file) => {
            Error(format!(
                "[time_series:04:{}:{}] Line {} file {}. Failed to parse date on line [{}].",
                code_file,
                code_line,
                line_num,
                file,
                line,
            ))
        },
        None => {
            Error(format!(
                "[time_series:04:{}:{}] Line {}. Failed to parse date on line [{}].",
                code_file,
                code_line,
                line_num,
                line,
            ))
        },
    }
}

///
pub fn parse_csv_value_failed(
    code_file: &str,
    code_line: u32,
    file: &Path,
    line_num: usize,
    line: &str) -> Error
 {
    match file.to_str() {
        
        // If possible to parse Path to str.
        Some(file) => {

            Error(format!(
                "[time_series:05:{}:{}] Line {} file {}. Failed to parse value on line [{}].",
                code_file,
                code_line,
                line_num,
                file,
                line,
            ))
        },
        None => {
            Error(format!(
                "[time_series:05:{}:{}] Line {}. Failed to parse value on line [{}].",
                code_file,
                code_line,
                line_num,
                line,
            ))
        },
    }
}

pub fn read_file_error(
    code_file: &str,
    code_line: u32,
    filename: &Path) -> Error
{
    match filename.to_str() {
        
        // If possible to parse Path to str.
        Some(file) => {
            Error(format!(
                "[time_series:06:{}:{}] Failed to read file [{}].",
                code_file,
                code_line,
                file, 
            ))
        },
        None => {
            Error(format!(
                "[time_series:06:{}:{}] Failed to read file.",
                code_file,
                code_line,
            ))
        },
    }
}


pub fn time_series_has_only_one_point(
    code_file: &str,
    code_line: u32) -> Error
{
    Error(format!(
        "[time_series:07:{}:{}] Time-series has only one point.",
        code_file,
        code_line,
    ))
}

///
pub fn time_series_is_empty(
    code_file: &str,
    code_line: u32) -> Error
{
    Error(format!(
        "[time_series:08:{}:{}] Time-series is empty.",
        code_file,
        code_line,
    ))
}
