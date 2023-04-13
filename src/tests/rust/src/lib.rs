#![no_std]
#![no_main]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use core::fmt::{Display, Write};
#[cfg(not(test))]
use core::panic::PanicInfo;

#[cfg(not(test))]
#[panic_handler]
pub extern "C" fn panic_handler(_: &PanicInfo) -> ! {
    // unsafe {
    //     printstr("Panic!!!")
    // }
    println("Panic!!!");
    loop {}
}

pub const MAX_MEM: u64 = 0x100000000;

extern "C" {
    pub fn hlt() -> !;
    pub(crate) fn printi_(rg: u64);
    pub(crate) fn printc_(rg: u8);
    pub fn llvm_memcpy_p0i8_p0i8_i64(dest: *mut u8, src: *const u8, n_bytes: u64);
    pub fn printptrln_asm(s: *const u8, len: usize);
}

pub fn printi(val: u64) {
    unsafe { printi_(val) }
}

pub fn printc(val: u8) {
    unsafe { printc_(val) }
}

#[doc(hidden)]
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn printptrln(s: *const u8, len: usize) {
    let s = core::slice::from_raw_parts(s, len);
    for c in s {
        printc(*c);
    }
    printc(b'\n');
}

pub fn println(s: &str) {
    unsafe { printptrln(s.as_bytes().as_ptr(), s.as_bytes().len()) }
}

pub struct PrintcWriter;

impl Write for PrintcWriter {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let s = s.as_bytes();
        for c in s {
            printc(*c);
        }

        Ok(())
    }

    fn write_fmt(&mut self, args: core::fmt::Arguments<'_>) -> core::fmt::Result {
        self.write_str(args.as_str().unwrap())
    }
}

#[repr(C, packed)]
pub struct BootInfo {
    pub max_mem: u64,
}

#[repr(C, packed)]
pub struct Foobar {
    pub pad0: [u8; 3],
    pub ptr: unsafe extern "C" fn(*const u8, usize),
}

unsafe impl Sync for Foobar {}

#[no_mangle]
pub fn kernel_main() -> ! {
    // unsafe extern "C" fn f(s: *const u8, len: usize) {
    //     printptrln_asm(s, len)
    // }
    // // let f = || println("Hello from a closure!");
    // let mut f = Foobar {
    //     pad0: [0; 3],
    //     ptr: f,
    // };
    // let s = b"Hello from a function pointer!\0";
    // f.pad0[2] = 0x69;
    // // printc(f.pad0[2]);
    // unsafe {
    //     (f.ptr)(s.as_ptr(), s.len());
    // }

    PrintcWriter.write_str("Hello from writeln!").unwrap();
    // panic!();
    unsafe { hlt() }
}
