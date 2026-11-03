"""One shared photograph renderer; no class names or answers are drawn."""
from PIL import Image,ImageDraw,ImageOps
import random
SIZE=(384,256)
def render(images,seed=0):
    rng=random.Random(seed)
    canvas=Image.new('RGB',SIZE,(245,246,248));draw=ImageDraw.Draw(canvas)
    for index,source in enumerate(images):
        x=(index%3)*128;y=(index//3)*128
        draw.rectangle((x+2,y+2,x+125,y+125),outline=(115,126,140),width=2)
        tile=ImageOps.contain(source.convert('RGB'),(112,112),Image.Resampling.BICUBIC)
        canvas.paste(tile,(x+(128-tile.width)//2,y+(128-tile.height)//2))
    return canvas
