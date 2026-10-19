require recipes-core/images/core-image-minimal.bb

SUMMARY = "Teaching image containing a timestamp contract smoke test"
IMAGE_INSTALL:append = " observation-check"
