extern fn printi(a: u64) void;
// extern fn printc(a: u8) void;
extern fn hlt() noreturn;
// extern fn und() noreturn;
// extern fn write_pt(pt: u64) noreturn;

pub export fn kernel_main() noreturn {
    printi(69696969);
    hlt();
}
