// const std = @import("std");
const builtin = @import("std").builtin;

export fn kernel_main() noreturn {
    printstrln_asm(@ptrCast(*const [:0]u8, "Hello From The Zig Kernel!"));
    hlt();
}

extern fn printi(a: u64) void;
extern fn printc(a: u8) void;
extern fn hlt() noreturn;
extern fn printstrln_asm(s: *const [:0]u8) void;

pub fn panic(msg: []const u8, stack_trace: ?*builtin.StackTrace) noreturn {
    _ = msg;
    _ = stack_trace;
    while (true) {}
}
