use mlirformat::Error;

use mlirpass::{Body, OpId, pass};
use super::{Xe3Image};

#[pass("xe3-pad", on = "builtin.module")]
pub fn xe3_pad<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    Xe3Image::pad(body, op);
    Ok(())
}
