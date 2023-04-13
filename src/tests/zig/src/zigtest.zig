// const std = @import("std");
const builtin = @import("std").builtin;

export fn addnums(a: i32, b: i32) i32 {
    try return a + b;
}

extern fn printi_(a: u64) void;
extern fn printc_(a: u8) void;

pub fn panic(msg: []const u8, stack_trace: ?*builtin.StackTrace) noreturn {
    _ = msg;
    _ = stack_trace;
    while (true) {}
}
