#pre-requisites: sudo apt-get install -y libnppicc11 libnppig11 libnppidei11 libnppif11
# In your docker-compose.yml, add to the Postiz backend service 
  environment:
  environment:
    - UPLOAD_MAX_FILE_SIZE=100MB
    - BODY_PARSER_LIMIT=100MB
    - EXPRESS_MAX_FILE_SIZE=104857600

