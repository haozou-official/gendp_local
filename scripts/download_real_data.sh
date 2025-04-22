mkdir -p data/real_aloha_demo

# download knife_real
curl -L https://uofi.box.com/shared/static/rkpiu70zej6tkckcqcbqeylsmdffi40u --output knife_real.zip
unzip knife_real.zip
rm knife_real.zip
mkdir -p data
mv knife_real data/real_aloha_demo
