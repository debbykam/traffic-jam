import requests

imageLink = "https://public.highwaystrafficcameras.co.uk/cctvpublicaccess/images/52660.jpg?sid=0.007634624891364128"
filename = "image_1.jpg"
# download image using GET
rawImage = requests.get(imageLink, stream=True)
# save the image received into the file
with open(filename, 'wb') as fd:
    for chunk in rawImage.iter_content(chunk_size=1024):
        fd.write(chunk)
