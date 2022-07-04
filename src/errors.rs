use std::error::Error;
use std::{fmt, path::PathBuf};

// ---ContiguousError-------------------------------------------------------------------------------

/// Data-values have inconsistent dates
#[derive(Debug)]
pub struct ContiguousError {
    pub line_num_opt: Option<u64>,
}

impl fmt::Display for ContiguousError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContiguousError({:?})", self.line_num_opt)
    }
}

impl Error for ContiguousError {}

// ---CsvError--------------------------------------------------------------------------------------

/// ?
#[derive(Debug)]
pub struct CsvError {
    pub data: String,
}

impl fmt::Display for CsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CsvError({})", self.data)
    }
}

impl Error for CsvError {}

// ---PartsError------------------------------------------------------------------------------------

/// Could not form a time-series from parts.
#[derive(Debug)]
pub struct PartsError;

impl fmt::Display for PartsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PartsError()")
    }
}

impl Error for PartsError {}

// ---IrregularError--------------------------------------------------------------------------------

/// CSV lines at `line_num` has length `current_len` but previous length was `prev_len`.
#[derive(Debug, PartialEq)]
pub struct IrregularError {
    pub line_num_opt: Option<u64>,
    pub prev_len: usize,
    pub current_len: usize,
}

impl fmt::Display for IrregularError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IrregularError({:?}, {}, {})",
            self.line_num_opt, self.prev_len, self.current_len
        )
    }
}

impl Error for IrregularError {}

// ---DateOrderError--------------------------------------------------------------------------------

/// `date1` was later than `date2`.
#[derive(Debug)]
pub struct DateOrderError {
    pub date1: String,
    pub date2: String,
}

impl fmt::Display for DateOrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DateOrderError({}, {})", self.date1, self.date2)
    }
}

impl Error for DateOrderError {}

// ---FilePathError---------------------------------------------------------------------------------

/// Could not find file at `path`.
#[derive(Debug)]
pub struct FilePathError {
    pub path: PathBuf,
}

impl fmt::Display for FilePathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FilePathError({})", self.path.display())
    }
}

impl Error for FilePathError {}

// ---EmptyError---------------------------------------------------------------------------------

#[derive(Debug)]
pub struct EmptyError();

impl fmt::Display for EmptyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EmptyError()")
    }
}

impl Error for EmptyError {}

// ---ParseDateError--------------------------------------------------------------------------------

/// An error that occurs if a date in CSV cannot be parsed. `ParseDateError(s, fmt)` where `s` is
/// the date string and `fmt` is the formatting string.
#[derive(Debug)]
pub struct ParseDateError {
    pub s: String,
    pub fmt: String,
}

impl fmt::Display for ParseDateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ParseDateError({}, {})", self.s, self.fmt)
    }
}

impl Error for ParseDateError {}

// ---ParseDataError--------------------------------------------------------------------------------
//

/// Data `field` at position `get` on `line_num` cannot be parsed.
#[derive(Debug)]
pub struct ParseDataError {
    pub line_num_opt: Option<u64>,
    pub s: String,
    pub get: usize,
}

impl fmt::Display for ParseDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ParseDataError({:?}, {}, {})",
            self.line_num_opt, self.s, self.get
        )
    }
}

impl Error for ParseDataError {}

// ---DataLenError----------------------------------------------------------------------------------

/// `DataLenError(line number, expected_len, found_len)`. Checked to see if length is
/// `expected_len` but actually `found_len`.
#[derive(Debug, PartialEq)]
pub struct DataLenError {
    pub line_num_opt: Option<u64>,
    pub expected_len: usize,
    pub found_len: usize,
}

impl fmt::Display for DataLenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DataLenError({:?}, {}, {})",
            self.line_num_opt, self.expected_len, self.found_len
        )
    }
}

impl Error for DataLenError {}

// ---FieldError-------------------------------------------------------------------------------

/// Tried to read a field that doesn't exist. `FieldError(len, get)` where `len` is the number
/// of fields and `get` is the field requested.
#[derive(Debug)]
pub struct FieldError {
    pub len: usize,
    pub get: usize,
}

impl fmt::Display for FieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FieldError({}, {})", self.len, self.get)
    }
}

impl Error for FieldError {}

// === Helper functions for impls =============================================================================

// /// Helper function for building `Date` impls.
// pub fn parse_date_err(data_str: &str, fmt: &str) -> Box<dyn Error> {
//     Box::new(ParseDateError(data_str.to_string(), fmt.to_string()))
// }
//
// // /// Helper function for building `Date` impls. An error if there are the wrong number of data
// // /// segment in a CSV record.
// // pub fn len_mismatch_err(record: &StringRecord, expected_len: usize) -> Box<dyn Error> {
// //     Box::new(LenError(record.as_string(), expected_len))
// // }
//
// pub fn parse_field_err(seg: &str) -> Box<dyn Error> {
//     Box::new(ParseFieldError(seg.to_string()))
// }
