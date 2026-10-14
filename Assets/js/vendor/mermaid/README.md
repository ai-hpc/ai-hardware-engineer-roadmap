# Mermaid runtime

This directory vendors the unmodified **Mermaid 11.17.2** standalone browser bundle.
It is loaded only on pages with diagrams, before Material for MkDocs, which keeps
responsibility for rendering and light/dark theme integration. Serving the runtime
with the site prevents a blocked or unavailable CDN from leaving raw diagram code.

- Upstream: [mermaid-js/mermaid](https://github.com/mermaid-js/mermaid)
- Package: [mermaid 11.17.2](https://www.npmjs.com/package/mermaid/v/11.17.2)
- Bundle source: [versioned distribution](https://cdn.jsdelivr.net/npm/mermaid@11.17.2/dist/mermaid.min.js)
- License: MIT; the adjacent `LICENSE` is from the same package. Bundled dependency
  notices remain intact at the end of the JavaScript file.
- SHA-256 (`mermaid-11.17.2.min.js`): `581ed7d74bd9048d0e3a91363927d72ef22942d7722546b27f7cc29e35390eb8`

To update, fetch a specific compatible version from the official package, retain
its license and bundled notices, update this record and the script path in
`overrides/main.html`, and build the site. Verify homepage and nested-page diagrams
with external requests blocked, including light and dark palettes. Do not replace
the pinned path with a floating version or add a second Mermaid initializer.
