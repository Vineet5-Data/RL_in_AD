# Makes src/ importable so `import preprocessing`/`import dataset`/etc. resolve
# under pytest. A conftest.py's mere presence anchors pytest's rootdir-based
# import-mode insertion (see pytest docs on rootdir/conftest discovery) -- this
# file's directory (src/) is what ends up on sys.path, no explicit code needed.
