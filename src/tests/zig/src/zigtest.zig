const std = @import("std");
const builtin = std.builtin;
const mem = std.mem;
const fmt = std.fmt;

extern fn printi(a: u64) void;
extern fn printc(a: u8) void;
extern fn hlt() noreturn;

fn println(s: []const u8) void {
    for (s) |c| {
        printc(c);
    }
    printc('\n');
}

export fn kernel_main(mem_size: u64) noreturn {
    println("Physical memory size is:");
    printi(mem_size);
    hlt();
}

pub fn panic(msg: []const u8, stack_trace: ?*builtin.StackTrace) noreturn {
    _ = stack_trace;
    var buf: [1024]u8 = undefined;
    const formatted = fmt.bufPrint(buf[0..], "Panic! {s}", .{msg}) catch "Panic!!!";
    println(formatted);
    while (true) {}
}
