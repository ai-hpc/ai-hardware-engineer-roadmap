MathJax 3.2.2: locally hosted browser components

The complete upstream es5/ distribution is copied here without modification.
The npm package LICENSE is retained as LICENSE. README.txt is course provenance.
This pins the site's existing MathJax v3 configuration; it is not a v4 migration.

Package: mathjax@3.2.2
Registry metadata: https://registry.npmjs.org/mathjax/3.2.2
Archive: https://registry.npmjs.org/mathjax/-/mathjax-3.2.2.tgz
Verified npm dist.integrity: sha512-Bt+SSVU8eBG27zChVewOicYs7Xsdt40qm4+UpHyX7k0/O9NliPc+x77k1/FEsPsjKPZGJvtRZM1vO+geW0OhGw==
Archive SHA-256: 1b9c0a1c44df864e915690558e72adb9cc5203360daefd385084ced3b6c64c09
Downloaded and verified UTC: 2026-10-19T00:22:47.048867+00:00
Local hosting instructions: https://docs.mathjax.org/en/v3.2/web/hosting.html

The page template loads tex-mml-chtml.js only for mathematical content.
Keep the component subdirectories and output/chtml/fonts/woff-v2 together so
component and font requests resolve on the same site without a CDN dependency.
