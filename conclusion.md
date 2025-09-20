—
Энэхүү орчуулга нь MIT лицензийн дагуу эх бүтээлээс хөрвүүлэв.
Эх сурвалж: Austin et al., "How to Scale Your Model" (https://jax-ml.github.io/scaling-book/)
Орч.: Mongolian (mn)
—

---
layout: distill
title: "Дүгнэлт ба Цаашдын Унших Ном"
# permalink: /main/
description: "Уншсанд баярлалаа! Энд бид цаашид судлахад хэрэгтэй хэдэн ном, эх сурвалжийг орууллаа."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

хэсгийн_дугаар: 11

previous_section_url: "../jax-stuff"
previous_section_name: "10-р хэсэг: JAX"

next_section_url: "../gpus"
next_section_name: "12-р хэсэг: GPU-ууд"

giscus_comments: үнэн

зохиогчид:
  - нэр: Жейкоб Остин
    url: "https://www.jacobaustin.org/"
    харьяалал:
      нэр: Google DeepMind
  - нэр: Шолто Дуглас
    url: "https://x.com/_sholtodouglas"
  - нэр: Рой Фростиг
    url: "https://cs.stanford.edu/~rfrostig/"
  - нэр: Ансельм Левская
    url: "https://anselmlevskaya.com/"
  - нэр: Чарли Чен
    url: "https://x.com/charliexychen"
  - нэр: Шарад Викрам
    url: "https://sharadvikram.com/"
  - нэр: Федерико Леброн
    url: "https://fedelebron.com/"
  - нэр: Питер Чой
    url: "https://x.com/pchoy95"
  - нэр: Винай Рамасеш
    url: "https://x.com/vinayramasesh"
  - нэр: Альберт Вебсон
    url: "https://representation.ai/"
  - нэр: Рейнер Попе<sup>*</sup>
    url: https://x.com/reinerpope

# Өөрийн бичлэгт агуулгын жагсаалт нэмэх.
#   - Агуулгын жагсаалтын нэрүүд нь тухайн хэсгийн нэртэй таарч байх ёстой,
#     ингэснээр бичлэг доторх холбоосууд зөв ажиллана.
#   - Доорх форматыг ашиглана уу, markdown агуулгын жагсаалтыг гараар бүү үүсгээрэй.
toc:
  - name: "Талархал"
  - name: "Цааш унших"
  - name: "Санал хүсэлт"

# Доор нэмэлт постод зориулсан тусгай стиль хэрхэн оруулах жишээ байна.
# Энэ нь энэ постын 'Layouts' хэсэгт ашиглагддаг.
# Хэрвээ та энэ постыг загвар болгон ашиглах бол энэ _styles блокийг устгаарай.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
  .algorithm {
    padding: 10px;
    margin-top: 5px;
    margin-bottom: 5px;
    border-style: dashed;
    background-color: #fffaf2;
  }

  .algorithm li {
    margin-bottom: 0px;
  }
---
**Энэ эссэний цувралыг уншсанд баярлалаа, мөн төгсгөлд нь хүрсэнд тань баяр хүргэе.** Дуусахаас өмнө, хэдэн талархал:

## Талархал

Энэ баримт бичиг нь Google DeepMind-ийн олон хүний ​​хамтын чухал хөрөнгө оруулалтыг илэрхийлж байна. Бид тэднийг товчхон дурдахыг хүсэж байна!

* Жеймс Брэдбюри, Рейнер Поп, болон Блэйк Хехтман энэ гар бичмэлийн олон санааг анх гаргаж, Transformer-ийн системийн талаарх ойлголтыг эрт олж авсан.
* Шолто Дуглас энэ баримтын анхны хувилбарыг бичиж, төслийг эхлүүлэхэд хариуцлагатай байсан. Тэрээр энэ баримтын ерөнхий өгүүлэмжийг хамгийн ихээр хариуцсан хүн юм.
* Жэйкоб Остин анхны хувилбарыг бүдүүн тэмдэглэлээс илүү боловсронгуй, бүрэн бүтэн баримт болгох ажлыг удирдсан. Тэрээр энэ баримтыг засах, форматлах, нийтлэх ажлын ихэнхийг хийж, бусад зохиогчдын хувь нэмрийг зохицуулсан.
* Ихэнх зураг болон хөдөлгөөнт зургийг Ансельм Левская болон Чарли Чен хийсэн.
* Чарли Чен inference хэсгийг бичиж, олон inference зураг зурсан.
* Рой Фростиг нийтлэх, засварлах болон бусад олон алхамд тусалсан.

Бид мөн энэ үйл явцын туршид чухал санал зөвлөгөө өгсөн олон хүмүүст талархаж байна. Тухайлбал, Zak Stone, Nikhil Sethi, Caitlin Stanton, Alex Dimitriev, Sridhar Lakshmanamurthy, Albert Magyar, Diwakar Gupta, Jeff Dean, Corry Wang, Matt Johnson, Peter Hawkins болон бусад олон хүнд баярлалаа. HTML форматлахад тусалсан Ruiqi Gao-д баярлалаа.

**Бүгдэд нь баярлалаа!**

<p markdown=1 class="announce">Явахынхаа өмнө, та бас NVIDIA GPU-ийн тухай шинэ [12-р хэсэг](../gpus)-ийг уншихыг сонирхож магадгүй!</p>

## Цааш унших

Дараахтай адил холбоотой бичвэрүүд байна:

* [**TPU Deep Dive**](https://henryhmko.github.io/posts/tpu/tpu.html): Энэ номын агуулгатай төстэйгээр TPU архитектурын талаар маш дэлгэрэнгүй, гайхалтай тайлбарласан нийтлэл.
* [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html): GPU болон PyTorch-д төвлөрсөн, LLM roofline болон performance engineering-ийн талаар илүү энгийн заавар.
* [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html): TPU програмчлал улам бүр өөрийн custom kernel-үүдийг Pallas ашиглан бичих шаардлагатай болж байна. Энэ цувралд kernel хэрхэн бичих болон энд дурдагдаагүй TPU-ийн доод түвшний олон зүйлийг тайлбарласан.
* [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM): Энэ нийтлэл GPU болон CUDA-д зориулагдсан боловч CUDA дээр matmul kernel хэрхэн optimize хийхийг маш сайн тайлбарласан. Энэ нь TPU болон GPU хэрхэн ялгаатайг ойлгоход тусална.
* [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html): JAX дээрх parallelism API-уудын талаар маш сайн гарын авлага бөгөөд энд ярьсан санаануудыг хэрхэн хэрэгжүүлэхийг сурахад тохиромжтой.
* [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024): Манай хуучин хамтрагч Rafi TPU performance engineering-ийн талаар маш сайн хичээл заасан бөгөөд бүх слайдууд GitHub дээр бий. Энэ нь эндээс илүү дэлгэрэнгүй олон зүйлийг хамарсан.
* [**\[2211.05102\] Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102): Transformer inference-ийн математикийн талаар дэлгэрэнгүй өгүүлэл. Энэ баримт бичгийн олон санааны эх үүсвэр болсон.
* [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook): Энэ номын GPU хувилбар мэт бөгөөд PyTorch дээр parallelism болон санах ой хэмнэх аргуудыг сургалтын үед хэрхэн хэрэгжүүлдгийг илүү дэлгэрэнгүй тайлбарласан.
* [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/): Энэ номтой төстэй олон санаа бүхий, сайн зураглалтай блог.
* [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework): LLM сургалт болон үйлчилгээний олон нарийн зүйлийг хамарсан гайхалтай Stanford-ийн курс. 1 болон 2-р даалгавар нь онцгой ач холбогдолтой.
* [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering): ML infrastructure-ийн талаар маш практик гарын авлага. Энэ номд дурдагдаагүй, үүлэн үйлчилгээ үзүүлэгчтэй хэлэлцээр хийх, кластер удирдах, GPU throughput хэмжих зэрэг сэдвүүдийг хамарсан.

Энэ салбарт бүрэн бичих боломж их байна, тиймээс бид энэ гар бичмэл илүү их бичихийг урамшуулна гэж найдаж байна! Мөн энэ бол судлах, шинжлэхэд үр дүнтэй салбар гэж бид итгэдэг. Олон тохиолдолд, олон hardware accelerator байхгүй байсан ч судалгаа хийж болно.

## Санал хүсэлт

Сайжруулахын тулд сэтгэгдэл эсвэл асуулт үлдээнэ үү. Манай зохиогч Жейкоб Остинтай jaaustin [at] google [dot] com хаягаар холбоо барьж болно. Мөн өөрчлөлт санал болгохыг хүсвэл асуудал үүсгэх, pull request илгээх эсвэл хэлэлцүүлэг хийх замаар [GitHub дээр](https://github.com/jax-ml/scaling-book) хандаарай.