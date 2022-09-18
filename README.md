## Inpaint
A wrapper of https://github.com/satoshiiizuka/siggraph2017_inpainting

https://user-images.githubusercontent.com/38597814/190922158-050934ff-edd4-41b7-bd08-fe6ae8a17ba3.mp4

### run server
Recommendation is running server is in docker. So, 
```bash
docker build --no-cache -t inpaint .
```
Then,
```bash
docker run --rm --net=host -it --gpus all inpaint:latest python3 inpaint/example/server.py
```
### run client (not interactive)
Install python package
```bash
pip3 install -e .
```
Then
```bash
python3 example/client.py
```
![out](https://user-images.githubusercontent.com/38597814/190915439-60dd0706-9555-47ba-a06a-1dd3b13a9703.png)


### run client (interactive)
The following code may reproduce the video above in the readme:
```bash
python3 example/interactive_client.py
```

