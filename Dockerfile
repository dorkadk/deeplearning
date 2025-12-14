# Használjunk egy hivatalos Python képet, ami elegendő a legtöbb DL projekthez.
# A 3.10-slim-buster egy könnyű, stabil alap.
# GPU támogatáshoz használj egy NVIDIA/CUDA alapú képet (pl. nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04)
# de a PyTorch és a GPU driverek verziójának egyeztetése kritikus!
FROM python:3.10-slim-buster

# Állítsuk be a munkakönyvtárat a konténerben
WORKDIR /app

# Másoljuk be a requirements fájlt a munkakönyvtárba
# Ez a sor segít a rétegek gyorsítótárazásában, mivel a függőségek ritkábban változnak.
COPY requirements.txt .

# Telepítsük a szükséges csomagokat
# A --no-cache-dir csökkenti a kép méretét
RUN pip install --upgrade pip && \
pip install --no-cache-dir -r requirements.txt

# Másoljuk be a teljes projekt kódot a munkakönyvtárba
# Feltételezzük, hogy a kódod a Dockerfile-lal azonos könyvtárban van
COPY src ./src



# Meghatározzuk, hogy mi induljon el a konténer indításakor.
# Ha csak interaktív futtatáshoz (pl. teszteléshez) használod, akkor ezt a sort kihagyhatod.
# Példa:
CMD ["python", "train_model.py"]