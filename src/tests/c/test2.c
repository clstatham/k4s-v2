int stringlen(char s[]) {
    int c = 0;
    while (s[c] != 0) {
        c++;
    }
    return c;
}

int foo() {
    return stringlen("Test String 1") + stringlen("Longer Test String 2");
}