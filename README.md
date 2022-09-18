## Inpaint
A wrapper of https://github.com/satoshiiizuka/siggraph2017_inpainting

![out](https://user-images.githubusercontent.com/38597814/190915439-60dd0706-9555-47ba-a06a-1dd3b13a9703.png)

### run server
Recommendation is running server is in docker. So, 
```bash
docker build --no-cache -t inpaint .
```
Then,
```bash
docker run --rm --net=host -it --gpus all inpaint:latest python3 inpaint/example/server.py
```
### run client
Install python package
```bash
pip3 install -e .
```
Then the following command outputs the figure in the top of the readme.
```bash
python3 example/client.py
```
