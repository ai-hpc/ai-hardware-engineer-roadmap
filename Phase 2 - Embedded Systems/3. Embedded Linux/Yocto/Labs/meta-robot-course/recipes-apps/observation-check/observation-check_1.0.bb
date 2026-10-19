SUMMARY = "Check observation age in an explicit millisecond clock domain"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=0304269c7842bb285234556ec250ca54"
SRC_URI = "file://observation-check.c file://LICENSE"
S = "${WORKDIR}"

do_compile() {
    ${CC} ${CPPFLAGS} ${CFLAGS} ${LDFLAGS} -std=c11 \
        "${S}/observation-check.c" -o "${B}/observation-check"
}

do_install() {
    install -d "${D}${bindir}"
    install -m 0755 "${B}/observation-check" "${D}${bindir}/observation-check"
}
