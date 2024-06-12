Steps to build docker agents


1. Update nvidia driver to at least 535 version

2. Build docker images

3. Modify `clearml.conf` file

  - Change credentials
  - Add docker agent image
  - Change `default_output_uri` for s3
  - Add S3 credentials
