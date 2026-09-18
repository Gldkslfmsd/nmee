

To download large data that are needed by scripts/our workflow, but are not committed:

`./download-large-data.sh`


To upload large data that should not be committed but are required by scripts or our worflow:

1. upload them anywhere on a public http(s) server so that it can be downloaded by wget/curl/cmdline tools
- an option that everybody has is an attachment of new release: https://github.com/Gldkslfmsd/nmee/releases/new 
2. add the download (or download+extract) script or commands as one section in `./download-large-data.sh`
