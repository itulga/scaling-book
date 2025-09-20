—
Энэхүү орчуулга нь MIT лицензийн дагуу эх бүтээлээс хөрвүүлэв.
Эх сурвалж: Austin et al., "How to Scale Your Model" (https://jax-ml.github.io/scaling-book/)
Орч.: Mongolian (mn)
—

# Моделоо хэрхэн томруулах вэ

Энэ ном нь LLM-уудыг TPU дээр өргөжүүлэх урлагийг ойлгомжтой болгох зорилготой. Бид TPU хэрхэн ажилладаг, LLM-уудыг том хэмжээнд хэрхэн ажиллуулдаг, сургалт болон таамаглал хийх үед харилцааны саадгүйгээр параллелизм схемийг хэрхэн сонгохыг тайлбарлахыг хичээсэн. Номыг https://jax-ml.github.io/scaling-book хаягаар унших боломжтой.

### Талархал

Энэ номыг Жэйкоб Остин, Шолто Дуглас, Рой Фростиг, Ансельм Левская, Чарли Чен, Шарад Викрам, Федерико Леброн, Питер Чой, Винай Рамасеш болон Альберт Вебсон нар Google DeepMind дээр бичсэн. Олон санаануудыг анх Жэймс Брэдбюри болон Рейнер Попе гаргасан.

Энэ вэбсайт нь https://github.com/alshedivat/al-folio болон Distill багийн бүтээсэн Distill-style Jekyll theme ашигладаг. Баярлалаа!

### Локал дээр ажиллуулах

Энэ репог өөрийн компьютерт суулгахын тулд танд Ruby, ImageMagick, болон Jupyter хэрэгтэй. Эдгээрийг MacOS дээр Homebrew ашиглан суулгаж болно.

```
brew install imagemagick ruby
pip install jupyter
```

Энэ суулгасны дараа, зөв Ruby хувилбар PATH-д байгаа эсэхийг шалгаарай. Танд хамгийн багадаа ruby 3.4.5 суулгасан байх хэрэгтэй. Та дараахыг нэмэх шаардлагатай байж магадгүй.

```
if [ -d "/opt/homebrew/opt/ruby/bin" ]; then
  export PATH=/opt/homebrew/opt/ruby/bin:$PATH
  export PATH=`gem environment gemdir`/bin:$PATH
fi
```

зөв хувилбарыг авахын тулд үүнийг .bashrc файлд нэмнэ үү. Үүний дараа та repository-г clone хийж, ажиллуулах боломжтой байх ёстой.

```
git clone https://github.com/jax-ml/scaling-book.git
cd scaling-book
bundle install
bundle exec jekyll serve
```

Та jekyll serve командыг амжилттай ажиллуулсны дараа ном `http://127.0.0.1:4000/scaling-book` дээр байх болно.

GitHub Pages сайт руу байрлуулахын тулд (repo-д бичих эрхтэй байх хэрэгтэй), `sh bin/deploy` командыг ажиллуулна уу. Энэ нь ойролцоогоор 3 минут үргэлжилнэ.

### Хамтран ажиллах ба Холбоо барих

Хэрвээ та ямар нэгэн асуудал харвал эсвэл асуулт байвал, вэбсайт дээр нь (Giscus ашигласан) сэтгэгдэл үлдээнэ үү эсвэл GitHub discussion-д бичээрэй. Хэрвээ та хувь нэмэр оруулахыг хүсвэл PR илгээж болно. Мөн jaaustin [at] google [dot] com хаягаар имэйл илгээж болно.

GitHub-д хувь нэмэр оруулахын тулд та Google-ийн "Contributor License Agreement" (CLA) буюу "Хувь нэмэр оруулагчийн лицензийн гэрээ"-нд гарын үсэг зурах хэрэгтэй. Та эндээс үүнийг хийж болно: https://cla.developers.google.com/clas.

### Ишлэл

Академик орчинд эшлэл хийхдээ энэ ажлыг дараах байдлаар ишлэнэ үү:

```Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.```

BibTeX citation

```
@article{scaling-book,
  title = {Таны Моделийг Хэрхэн Томруулах вэ},
  author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
  publisher = {Google DeepMind},
  howpublished = {Онлайн},
  note = {https://jax-ml.github.io/scaling-book/ хаягаас авсан},
  year = {2025}
}

![луус](assets/img/dragon.png)

*Энэ номыг анх "How To Scale Your Dragon" гэж нэрлэсэн. Энэ нь Dreamworks киноноос санаа авсан бөгөөд тиймээс л лууны зураглал орсон юм.*