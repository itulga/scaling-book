—
Энэхүү орчуулга нь MIT лицензийн дагуу эх бүтээлээс хөрвүүлэв.
Эх сурвалж: Austin et al., "How to Scale Your Model" (https://jax-ml.github.io/scaling-book/)
Орч.: Mongolian (mn)
—

---
layout: distill
title: "TPU-ний тухай хэрхэн бодох вэ"
# permalink: /main/
description: "Энэ хэсэгт TPU хэрхэн ажилладаг, олон чипийг хэрхэн холбож сургалт болон inference хийх талаар, мөн энэ нь бидний дуртай алгоритмуудын гүйцэтгэлд хэрхэн нөлөөлдөг талаар ярилцана. GPU хэрэглэгчдэд ч бас хэрэгтэй мэдээлэл бий!"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

# Илгээхдээ нэрээ нууцал

хэсгийн_дугаар: 2

previous_section_url: "../roofline"
previous_section_name: "1-р хэсэг: Rooflines"

next_section_url: ../sharding
next_section_name: "3-р хэсэг: Шардинг"

ном зүй: main.bib

giscus_comments: true

authors:
  - name: Жейкоб Остин
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Шолто Дуглас
    url: "https://x.com/_sholtodouglas"
  - name: Рой Фростиг
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Ансельм Левская
    url: "https://anselmlevskaya.com/"
  - name: Чарли Чен
    url: "https://x.com/charliexychen"
  - name: Шарад Викрам
    url: "https://sharadvikram.com/"
  - name: Федерико Леброн
    url: "https://fedelebron.com/"
  - name: Питер Чой
    url: "https://x.com/pchoy95"
  - name: Винай Рамасеш
    url: "https://x.com/vinayramasesh"
  - name: Альберт Вебсон
    url: "https://representation.ai/"
  - name: Райнер Попе<sup>*</sup>
    url: https://x.com/reinerpope

# Өөрийн бичлэгт агуулгын жагсаалт нэмэх.
#   - Агуулгын жагсаалтын нэрүүд нь тухайн хэсгийн нэртэй яг таарч байх ёстой
#     ингэснээр бичлэг доторх холбоосууд зөв ажиллана.
#   - Энэ форматыг ашиглана уу, гар аргаар markdown агуулгын жагсаалт бүү үүсгээрэй.
toc:
  - name: TPU гэж юу вэ?
  - name: TPU сүлжээ
  - name: Гол санаанууд
  - subsections:
    - name: TPU техникийн үзүүлэлтүүд
  - name: Дасгалтай бодлогууд
  - name: Хавсралт
  - subsections:
    - name: "Хавсралт A: TPU дотоод бүтэц илүү дэлгэрэнгүй"
    - name: "Хавсралт B: Систолик массив хэрхэн ажилладаг вэ?"

# Доор нэмэлт постод зориулсан тусгай стиль хэрхэн оруулах жишээ байна.
# Энэ нь энэ постын 'Layouts' хэсэгт ашиглагддаг.
# Хэрвээ та энэ постыг загвар болгон ашиглах бол, энэ _styles блокийг устгаарай.
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
---

<p markdown=1 class="announce">Танд NVIDIA GPU-үүдийн тухай шинэ [12-р хэсэг](../gpus)-ийг унших бас сонирхолтой байж магадгүй!</p>

## TPU гэж юу вэ?

**TPU гэдэг нь үндсэндээ матриц үржүүлэхэд (TensorCore гэж нэрлэдэг) зориулсан тооцооллын цөм бөгөөд хурдан санах ойтой (өндөр зурвасын өргөнтэй санах ой буюу HBM гэж нэрлэдэг) холбогдсон байдаг<d-cite key="tpu_paper"></d-cite>.** Энэ бол диаграмм:

{% include figure.liquid path="assets/img/tpu-chip.png" class="img-fluid" caption="<b>Зураг:</b> TPU чипийн үндсэн бүрдэл хэсгүүд. TensorCore нь саарал зүүн талын хайрцаг бөгөөд matrix-multiply unit (MXU), vector unit (VPU), болон vector memory (VMEM) агуулна." %}

TensorCore-ыг үндсэндээ маш сайн матриц үржүүлэх машин гэж ойлгож болно, гэхдээ үүнээс гадна анхаарах хэдэн өөр функцтэй. TensorCore нь гурван гол нэгжтэй:

* **MXU** (Matrix Multiply Unit) нь TensorCore-ийн гол хэсэг юм. Ихэнх TPU үеүүдэд энэ нь нэг `bfloat16[8,128] @ bf16[128,128] -> f32[8,128]` матриц үржүүлэх үйлдлийг <d-footnote>TPU v6e (Trillium) нь 256x256 MXU-тай, харин өмнөх бүх үеүүд нь 128x128 ашигладаг</d-footnote> 8 цикл тутамд хийдэг бөгөөд энэ нь systolic array ашигладаг (дэлгэрэнгүйг <a href="#appendix-b-how-does-a-systolic-array-work">B хавсралт</a>-аас үзнэ үү).
  * Энэ нь TPU v5e дээр 1.5GHz хурдтай үед нэг MXU-д `5e13` bf16 FLOPs/сек орчим байна. Ихэнх TensorCore-д 2 эсвэл 4 MXU байдаг, жишээ нь TPU v5e-ийн нийт bf16 FLOPs/сек нь `2e14` юм.
  * TPU нь мөн илүү хурдан ажиллахын тулд бага нарийвчлалтай матриц үржүүлэхийг дэмждэг (жишээ нь, нэг TPU v5e чип нь `4e14` int8 OPs/сек хийж чадна).

* **VPU** (Вектор боловсруулах нэгж) нь ерөнхий математик үйлдлүүдийг хийдэг. Жишээ нь, ReLU идэвхжүүлэлт, эсвэл векторуудын хооронд нэмэх, үржүүлэх зэрэг үйлдлүүд. Мөн нийлбэр (reduction) үйлдлүүдийг энд хийдэг. <a href="#appendix-a-more-on-tpu-internals">A хавсралт</a> хэсэгт дэлгэрэнгүй мэдээлэл бий.
* **VMEM** (Вектор санах ой) нь чипэн дээрх түр хадгалах сан бөгөөд TensorCore дотор, тооцооллын нэгжүүдийн ойролцоо байрладаг. Энэ нь HBM-ээс хамаагүй жижиг (жишээ нь, TPU v5e дээр 128 MiB) боловч MXU-д илүү хурдан өгөгдөл дамжуулдаг. VMEM нь CPU дээрх L1/L2 кэштэй төстэй боловч илүү том бөгөөд программист өөрөө удирддаг. HBM доторх өгөгдлийг эхлээд VMEM рүү хуулж байж TensorCore тооцоолол хийж чадна.

**TPU-ууд матриц үржүүлэхэд маш, маш хурдан байдаг**. Энэ бол тэдний гол ажил бөгөөд тэд үүнийг сайн хийдэг. [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) нь одоогийн хамгийн хүчирхэг TPU-уудын нэг бөгөөд `2.5e14` bf16 FLOPs / секунд / цөм эсвэл `5e14` bf16 FLOPs / сек / чип хийх чадвартай. 8960 чиптэй нэг pod нь 4 эксафлопс / секунд хийж чадна. Энэ бол *маш их* гэсэн үг. Энэ нь дэлхийн хамгийн хүчирхэг суперкомпьютеруудын нэг юм. Google иймэрхүү олон TPU-тэй.<d-footnote>TPU-ууд, ялангуяа тэдний systolic array-ууд нь маш хүчтэй hardware accelerator болдог. Учир нь матриц үржүүлэх нь цөөн хэдэн алгоритмуудын нэг бөгөөд $O(n^3)$ тооцоололыг $O(n^2)$ байт дээр ашигладаг. Энэ нь энгийн ALU-г тооцооллоор удаашруулахад амархан болгож, санах ойн bandwidth-аар биш.</d-footnote>

Дээрх зурагт бас SMEM болон scalar unit зэрэг хэдэн өөр бүрэлдэхүүн орсон байна. Эдгээрийг control flow-ийг удирдахад ашигладаг ба <a href="#appendix-a-more-on-tpu-internals">A хавсралт</a>-д товч тайлбарласан, гэхдээ эдгээрийг заавал ойлгох шаардлагагүй. Харин HBM бол чухал бөгөөд ойлгоход амархан:

* **HBM** (Өндөр зурвасын өргөнтэй санах ой) нь TensorCore ашиглахад зориулсан тэнцэрүүдийг хадгалдаг хурдан санах ойн том хэсэг юм. HBM нь ихэвчлэн хэдэн арван гигабайтын багтаамжтай байдаг (жишээ нь, [TPU v5e нь 16GiB HBM-тэй](https://cloud.google.com/tpu/docs/v5e#system_architecture)).

  * Тооцоололд хэрэгтэй үед, тэнзоруудыг HBM-ээс VMEM-рүү (доор үзнэ үү) дамжуулж MXU-д оруулна, үр дүнг VMEM-ээс буцааж HBM-д бичнэ.

  * HBM ба TensorCore (VMEM-ээр дамжуулан) хоорондын дамжуулах хурдыг "HBM дамжуулах хурд" гэж нэрлэдэг (ихэвчлэн 1-2TB/сек орчим байдаг) бөгөөд энэ нь санах ойд хамааралтай ажлуудыг хэр хурдан тооцоолохыг хязгаарладаг.

**Ерөнхийдөө, бүх TPU үйлдлүүд дамжуулах хоолой шиг (pipelined) ба давхарлаж (overlapped) хийгддэг.** Матриц үржүүлэх (matmul) үйлдэл хийхийн тулд TPU эхлээд матрицын хэсгүүдийг $A$ ба $X$ HBM-ээс VMEM рүү хуулна, дараа нь эдгээрийг MXU-д ачаалж, 8x128 (for $X$) болон 128x128 (for $A$) хэмжээтэй хэсгүүдийг үржүүлнэ, дараа нь үр дүнг хэсэг хэсгээр нь HBM рүү буцааж хуулна. Үүнийг үр дүнтэй хийхийн тулд matmul үйлдлийг дамжуулах хоолой шиг (pipelined) болгож, VMEM рүү/VMEM-ээс хуулж буй үйлдлийг MXU-ийн ажилтай давхардуулдаг (overlapped). Ингэснээр MXU үргэлжлүүлэн ажиллах боломжтой болж, санах ойн дамжуулалтыг хүлээхгүй, matmul үйлдэл тооцооллоороо хязгаарлагдаж, санах ойгоороо хязгаарлагдахгүй.

Энд HBM-ээс элемент бүрээр үржүүлэх үйлдлийг хэрхэн хийх жишээг харууллаа:

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>Зураг:</b> TPU дээр pointwise product хэрхэн хийгдэж байгааг харуулсан хөдөлгөөнт зураг. HBM-ээс byte-уудыг ачаалж байна. Byte-уудыг санах ойгоос хэсэг хэсгээр нь уншиж, хэсэгчилсэн үр дүнгүүдийг бүх array бүрэн ачаалагдахаас өмнө шууд дамжуулж буцааж байна." %}

Матмул бараг адилхан харагдана, гэхдээ энэ нь VPU/Vector unit-ийн оронд MXU-д ачаалагдана, мөн өгөгдлийг ачаалах ба хадгалах дараалал өөр байна, учир нь нэг жингийн хэсгийг олон идэвхжүүлэлтийн хэсэгт ашигладаг. Та өгөгдлийн хэсгүүд VMEM рүү урсаж байгааг, дараа нь VREGs (вектор бүртгэгч)-т, дараа нь Vector Unit рүү, дараа нь буцаад VMEM болон HBM рүү очиж байгааг харж болно. Одоо бид харах гэж байна: хэрвээ HBM-ээс VMEM рүү ачаалах хурд Vector Unit (эсвэл MXU)-ийн FLOPs-оос удаан бол бид "bandwidth bound" буюу дамжуулах зурвасын өргөнд хязгаарлагдсан болно, учир нь бид VPU эсвэл MXU-д хийх ажил дутагдаж байна.

<p markdown=1 class="takeaway">**Гол санаа:** TPU нь маш энгийн. Тэд HBM-ээс VMEM рүү жингүүдийг ачаалдаг, дараа нь VMEM-ээс systolic array руу дамжуулдаг. Systolic array нь секундэд ойролцоогоор 200 их наяд үржүүлж-нэмэх үйлдэл хийж чадна. HBM $\leftrightarrow$ VMEM болон VMEM $\leftrightarrow$ systolic array-ийн дамжуулах хурд нь TPU-ууд ямар тооцооллыг үр дүнтэй хийж чадахыг үндсэнд нь хязгаарладаг.</p>

**VMEM ба арифметик эрчимшил:** VMEM нь HBM-ээс хамаагүй жижиг боловч MXU-д илүү өндөр дамжуулах зурвастай. [1-р хэсэгт](../roofline) бид харсанчлан, хэрвээ алгоритм нь бүх оролт/гаралтыг VMEM-д багтааж чадвал, харилцааны сааталд өртөх магадлал бага байна. Энэ нь арифметик эрчимшил муутай тооцоололд их хэрэгтэй: VMEM-ийн зурвас нь HBM-ээс ойролцоогоор 22 дахин их тул, VMEM-ээс унших/бичих MXU үйлдэл хамгийн их FLOPs ашиглахын тулд ердөө 10-20 арифметик эрчимшил шаардлагатай. Энэ нь бид жингүүдээ HBM-д биш VMEM-д багтааж чадвал, матриц үржвэрүүд маань маш бага batch хэмжээтэй үед FLOPs-аар хязгаарлагдах боломжтой гэсэн үг. Мөн үндсэндээ бага арифметик эрчимшилтэй алгоритмууд ч үр дүнтэй байж чадна. Гэхдээ VMEM маш жижиг тул энэ нь ихэвчлэн хүндрэл болдог.<d-footnote>Бид заримдаа VMEM урьдчилан ачаалах (prefetching) тухай ярьдаг, энэ нь матмул хийхийн өмнө жингүүдээ VMEM-д урьдчилан ачаалж, ачаалах зардлыг нуухыг хэлдэг. Жишээ нь, энгийн Transformer-д бид заримдаа том feed-forward жингүүдээ анхаарал (attention) үеийн үед VMEM-д ачаалж болдог, ингэснээр санах ойн зурвасаар хязгаарлагдсан үед жинг ачаалах зардлыг нууж чадна. Үүний тулд жингүүд маань хангалттай жижиг эсвэл хэсэглэгдсэн (sharded) байх ёстой, ингэснээр нэг давхаргыг VMEM-д багтааж, нэмэлт зай үлдээж чадна.</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

**TPU чип ихэнхдээ (гэхдээ үргэлж биш) хоёр TPU цөмөөс бүрддэг бөгөөд эдгээр нь санах ойгоо хуваалцдаг ба нэг том хурдасгуур гэж бодож болно**. Энэ тохиолдолд FLOPs хоёр дахин их байдаг (үүнийг "megacore" тохиргоо гэж нэрлэдэг). Энэ нь TPU v4-өөс хойш үнэн болсон. Хуучин TPU чипүүд тусдаа санах ойтой бөгөөд хоёр тусдаа хурдасгуур гэж үздэг (TPU v3 болон өмнөх хувилбарууд). Inference-д зориулсан чипүүд, жишээ нь TPU v5e, зөвхөн нэг TPU цөмтэй байдаг.

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**Чипүүд** нь **4-ийн багцаар ‘таваг’ дээр** байрладаг бөгөөд **CPU host-д PCIe сүлжээгээр** холбогддог. Энэ бол ихэнх уншигчдад танил формат юм: 4 чип (8 цөм, гэхдээ ихэвчлэн 4 логик мегацөм гэж үздэг) нь Colab эсвэл нэг TPU-VM-ээр ил гардаг. TPU v5e шиг inference чипүүдэд бид нэг host-д 2 тавагтай байдаг, гэхдээ чип бүрт зөвхөн 1 цөмтэй, тэгэхээр 8 чип = 8 цөм болно.<d-footnote>Cloud TPU VM дээр таваг бүр тусдаа VM-ийн нэг хэсэг болж ил гардаг тул дахин 4 цөм харагдана.</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe дамжуулах хурд хязгаартай:** HBM $\leftrightarrow$ VMEM холбоосын адил, CPU $\leftrightarrow$ HBM PCIe холболт нь тодорхой дамжуулах хурдтай бөгөөд энэ нь хост санах ойгоос HBM руу эсвэл эсрэгээр нь өгөгдөл ачаалах хурдыг хязгаарладаг. Жишээ нь, TPU v4-ийн PCIe дамжуулах хурд нь хоёр чиглэлд тус бүр 16GB / секунд байдаг бөгөөд энэ нь HBM-ээс бараг 100 дахин удаан гэсэн үг юм. Бид өгөгдлийг хост (CPU) RAM руу ачаалж/буулгаж чадна, гэхдээ маш хурдан биш.

## TPU Сүлжээ

**Чипүүд нь Pod дотор ICI сүлжээгээр холбогддог**. Хуучин үеийн (TPU v2 болон TPU v3), inference чипүүд (жишээ нь, TPU v5e), болон Trilium (TPU v6e) дээр ICI ("inter-chip interconnects" буюу чип хоорондын холболт) нь хамгийн ойр 4 хөрш чипийг холбодог (ирмэгийн холбоосоор 2D torus үүсгэнэ). TPU v4 болон TPU v5p нь хамгийн ойр 6 хөрш чиптэй холбогддог (3D torus үүсгэнэ). Эдгээр холболтууд нь **эзэмшигч (host)**-ээр дамждаггүй, чипүүдийн хооронд шууд холбогддог.

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

Тороид бүтэц нь ямар ч хоёр node хоорондын хамгийн их зайг $N$-аас $N / 2$ болгож багасгадаг бөгөөд ингэснээр харилцаа холбоо илүү хурдан болдог. TPU-ууд мөн "эргэлдсэн торус" (twisted torus) гэсэн тохиргоотой бөгөөд энэ нь торусыг Möbius-strip шиг топологиор ороож, node-уудын дундаж зайг цааш нь багасгадаг.

**TPU pod-ууд (ICI-ээр холбогдсон) маш том болж чадна:** хамгийн их pod-ийн хэмжээ (үүнийг **superpod** гэж нэрлэдэг) нь TPU v4-д `16x16x16`, TPU v5p-д `16x20x28` байна. Эдгээр том pod-ууд нь `4x4x4` chip-үүдийн дахин тохируулж болдог шооноос бүрддэг бөгөөд [оптик wraparound холбоос](https://arxiv.org/pdf/2208.10041)<d-footnote>Оптик switch нь зүгээр л дахин тохируулж болдог холболт бөгөөд яг ижил ICI bandwidth-тай. Энэ нь бидэнд шоо холбох боломжийг олгодог ба wraparound холбоосыг хадгалдаг.</d-footnote>-оор холбогдсон байдаг. Бид эдгээрийг дахин тохируулж маш том topology-уудыг холбож чадна.

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

Жижиг топологиуд (жишээ нь, `2x2x1`, `2x2x2`) мөн хүсэж болно, гэхдээ wraparound-гүй байна. Энэ нь чухал анхааруулга, учир нь ихэнх харилцааны хугацаа ихэвчлэн хоёр дахин нэмэгддэг. Бүтэн кубын аль ч үржвэр (жишээ нь, `4x4x4` эсвэл `4x4x8`) нь wraparound-уудыг оптик унтраалгаар хангана.<d-footnote>Анхаарна уу: `2x2x4` нь wraparound-гүй байна, учир нь эдгээрийг зөвхөн бүтэн куб дээр байдаг оптик унтраалгаар хангадаг. TPU v5e 8x16 нь урт тэнхлэг дээрээ wraparound-тай байна, учир нь энэ нь дахин тохируулах боломжтой оптик сүлжээ ашигладаггүй.</d-footnote>

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e болон Trillium pod-ууд нь нэг `16x16` 2D торус-с бүрддэг бөгөөд аливаа тэнхлэг дээр 16 хэмжээтэй бол wraparound буюу эргэлттэй байдаг (энэ нь `8x16` нь урт тэнхлэг дээр wraparound-тай гэсэн үг). TPU v5e болон v6e (Trillium) нь 16x16 торус-оос том болж чадахгүй, гэхдээ pod-ууд нь стандарт дата-центр сүлжээ (DCN)-ээр дамжуулан хоорондоо харилцаж чадна. DCN нь TPU host-уудыг хооронд нь холбодог. Мөн, жижиг топологи хүсвэл dims $<16$ дээр wrap байхгүйгээр захиалж болно.

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

**Энэ хамгийн ойрын-сайтан холболт нь TPU ба GPU-гийн гол ялгаа юм**. GPU-ууд нь шаталсан унтраалга (switch)-аар холбогдсон байдаг бөгөөд энэ нь бүх GPU хооронд шууд холболттой мэт ажилладаг, харин TPU шиг орон нутгийн холболт ашигладаггүй. Ихэнхдээ, нэг node доторх GPU-ууд (H100-д 8 GPU эсвэл B200 NVL72-д 72 хүртэл GPU) шууд холбогдсон байдаг, харин том топологи дээр GPU бүрийн хооронд O(log(N)) удаа дамжих шаардлагатай болдог. Нэг талаас, энэ нь GPU-ууд цөөн дамжуулалтаар дурын өгөгдөл илгээж чадна гэсэн үг. Нөгөө талаас, TPU-ууд маш хямд (учир нь NVLink унтраалга үнэтэй), холбоход энгийн, мөн илүү том топологи руу өргөжих боломжтой байдаг, учир нь төхөөрөмж бүрийн холболтын тоо болон зурвасын өргөн тогтмол байдаг. Дэлгэрэнгүйг [энд](../gpus#networking) уншаарай.

**ICI нь DCN-ээс харьцангуй хурдан боловч HBM-ийн дамжуулах хурднаас удаан байна.** Жишээ нь, [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) нь:

* `2.5e12` байт/сек (2.5 ТБ/сек) HBM дамжуулах хурд нэг чип бүрт.
* `9e10` байт/сек (90 ГБ/сек) ICI дамжуулах хурд нэг тэнхлэг бүрт, нэг чип дээр 3 тэнхлэгтэй.<d-footnote>Дээрх хуудсанд 100 ГБ/сек дамжуулах хурд гэж бичсэн байгаа, энэ нь энд бичсэнээс бага зэрэг өөр байна. TPU ICI холболтуудын дамжуулах хурд нь хийж буй үйлдлээсээ хамаараад бага зэрэг өөр байдаг. Та ерөнхийдөө энэ баримт бичиг дээрх тоог санаа зоволгүй ашиглаж болно.</d-footnote>
* `6.25e9` байт/сек (6.25 ГБ/сек) DCN (гаралт) дамжуулах хурд нэг TPU бүрт (тус бүрийн хост дээр 1-2 NIC-ээр дамжуулна).<d-footnote>TPU v6e нь 12.5e9 байт/сек, v5e нь 3.125e9 байт/сек дамжуулах хурдтай.</d-footnote>

Энэ нь бид загваруудыг олон чип дээр хуваах үед, удаан төхөөрөмж хоорондын холбооноос болж MXU-г удаашруулахгүй байхыг анхаарах хэрэгтэй гэсэн үг юм.

**Олон зүсэлттэй сургалт:** ICI-ээр холбогдсон TPU-уудын багцыг **зүсэлт** гэж нэрлэдэг. Өөр өөр зүсэлтүүдийг DCN ашиглан хооронд нь холбож болно, жишээ нь өөр өөр pod-ууд дээрх зүсэлтүүдийг холбох. DCN нь ICI-ээс хамаагүй удаан холболт тул бидний тооцоолол DCN-ээс өгөгдөл хүлээх хугацааг багасгахыг хичээх хэрэгтэй. DCN нь host-оос host руу холболт тул TPU-аас TPU руу DCN-ээр buffer дамжуулахын тулд эхлээд PCIe-ээр host руу дамжуулна, дараа нь сүлжээгээр гаргана, дараа нь зорилтот host сүлжээгээр орж, эцэст нь PCIe-ээр HBM руу дамжуулна.

## Гол санаанууд

* TPU-ууд энгийн бөгөөд ихэнх тохиолдолд санах ойд (маш хурдан) холбогдсон матриц үржих нэгж гэж ойлгож болно, бусад чипүүдтэй ICI-ээр (нэлээд хурдан), дата төвийн бусад хэсэгтэй DCN-ээр (харьцангуй хурдан) холбогддог.

* Харилцаа холбоо нь бидний сүлжээний дамжуулах чадвараар хязгаарлагддаг. Дараах дарааллаар хурдтай байна:
  * HBM дамжуулах чадвар: TensorCore болон түүнд холбогдсон HBM-ийн хооронд.
  * ICI дамжуулах чадвар: TPU чип болон хамгийн ойрын 4 эсвэл 6 хөрш чипийн хооронд.
  * PCIe дамжуулах чадвар: CPU host болон түүнд холбогдсон чипүүдийн tray(үүд)-ийн хооронд.
  * DCN дамжуулах чадвар: Олон CPU host-уудын хооронд, ихэвчлэн ICI-ээр холбогдоогүй host-уудын хооронд.

* **Слайсын дотор TPU-ууд зөвхөн хамгийн ойрын хөршүүдтэйгээ ICI-ээр холбогдсон байдаг.** Энэ нь слайсын доторх холын чипүүдийн хооронд ICI-ээр харилцахын тулд дундын чипүүдээр дамжин өнгөрөх хэрэгтэй гэсэн үг юм.

* **Жингийн матрицуудыг хамгийн багадаа 128 хэмжээтэй болтол дүүргэх хэрэгтэй** (TPU v6 дээр 256) хоёр чиглэлд хоёуланд нь MXU-г бүрэн ашиглахын тулд (үнэндээ, бага хэмжээтэй талыг 128 хүртэл дүүргэнэ).

* **Бага нарийвчлалтай matrix multiplication ихэвчлэн хурдан байдаг.** TPU-ууд int8 эсвэл int4 FLOP-уудыг bfloat16 FLOP-оос ойролцоогоор 2x/4x хурдан хийж чаддаг (үүнийг дэмждэг үеүүдэд). VPU үйлдлүүд одоо ч fp32 дээр хийгддэг.

* TPU тооцоолох нэгжид саатал үүсэхээс сэргийлэхийн тулд бид **сувгийн хурдтай харьцуулахад мэдээлэл дамжуулалтын хэмжээг тохируулах** хэрэгтэй.

### TPU-ийн Үзүүлэлтүүд

Манай чипүүдийн зарим тодорхой тоонууд энд байна:

| Загвар                                       | Pod хэмжээ | Host хэмжээ | HBM багтаамж/чип | HBM BW/чип (байт/сек) | FLOPs/сек/чип (bf16) | FLOPs/сек/чип (int8) |
| :----------------------------------------- | :--------: | :--------: | :--------------: | :-------------------: | :------------------: | :------------------: |
| <span class="nowrap-header">TPU v3</span>  |   32x32    |   4x2      |      32GB        |       9.0e11          |      1.4e14          |      1.4e14          |
| <span class="nowrap-header">TPU v4p</span> | 16x16x16   |  2x2x1     |      32GB        |       1.2e12          |      2.75e14         |      2.75e14         |
| <span class="nowrap-header">TPU v5p</span> | 16x20x28   |  2x2x1     |      96GB        |       2.8e12          |      4.59e14         |      9.18e14         |
| <span class="nowrap-header">TPU v5e</span> |   16x16    |   4x2      |      16GB        |       8.1e11          |      1.97e14         |      3.94e14         |
| <span class="nowrap-header">TPU v6e</span> |   16x16    |   4x2      |      32GB        |       1.6e12          |      9.20e14         |      1.84e15         |

Host size гэдэг нь нэг host-д холбогдсон TPU-уудын topology-г хэлнэ (жишээ нь, TPU v5e нь нэг CPU host-оор 8 TPU-г 4x2 topology-оор холбосон байдаг). Энд interconnect-ийн тоонууд байна:

| Загвар       | ICI BW/шугам (нэг чиглэл, bytes/s) | ICI BW/шугам (хоёр чиглэл, bytes/s) |
| :----------- | :-------------------------------: | :-------------------------------: |
| **TPU v3**   |              1e11                 |             2e11                  |
| **TPU v4p**  |             4.5e10                |             9e10                  |
| **TPU v5p**  |              9e10                 |            1.8e11                 |
| **TPU v5e**  |             4.5e10                |             9e10                  |
| **TPU v6e**  |              9e10                 |            1.8e11                 |

Бид нэг чиглэлтэй (unidirectional) зурвасын өргөн болон хоёр чиглэлтэй (bidirectional, эсвэл bidi) зурвасын өргөнийг хоёуланг нь оруулсан. Учир нь нэг чиглэлтэй зурвасын өргөн нь техник хангамжид илүү үнэн байдаг, гэхдээ хоёр чиглэлтэй зурвасын өргөн нь бүтэн ring-тэй холбоотой тэгшитгэлүүдэд илүү олон удаа тохиолддог.<d-footnote>Бид хоёр чиглэлтэй (bidirectional, эсвэл bidi) зурвасын өргөн гэж нэг холбоосоор хоёр чиглэлд дамжуулж болох нийт byte-ыг хэлж байна, эсвэл өөрөөр хэлбэл, нэг TPU-аас тодорхой тэнхлэг дагуу гарч болох нийт byte-ыг хэлж байна. Энэ нь бид хоёр холбоосыг үр дүнтэй ашиглаж чадвал үнэн байдаг. Энэ нь ring зөв ажиллаж байвал, өөрөөр хэлбэл тухайн тэнхлэг дээр wraparound холболт байгаа үед үнэн. Энэ нь inference chip дээр бүтэн 16 тэнхлэгтэй үед, эсвэл training chip (v*p) дээр тухайн тэнхлэг нь 4-ийн үржвэр байх үед тохиолддог. Бид ихэвчлэн хоёр чиглэлтэй зурвасын өргөнийг ашиглахыг илүүд үздэг, учир нь энэ нь хоёр чиглэлтэй холбоо бүхий тооцоололд олон удаа гарч ирдэг.</d-footnote>

PCIe-ийн дамжуулах чадал ихэвчлэн нэг TPU тутамд `1.6e10` байт / секунд байдаг (TPU v6e-д `3.2e10`), харин DCN-ийн дамжуулах чадал ихэвчлэн нэг TPU тутамд `6.25e9` байт / секунд байдаг (TPU v6e-д `12.5e9`, TPU v5e-д `3.125e9`).

## Ажилласан бодлогууд

Эдгээр тоонууд жаахан уйтгартай байж болох ч тэдгээр нь загварын гүйцэтгэлийг үндсэн roofline тооцоолол хийх боломж олгодог. Яагаад энэ нь хэрэгтэй болохыг тайлбарлахын тулд хэдэн бодлого бодъё. Илүү олон жишээг 3-р хэсэгт үзэх болно.

**Асуулт 1 [LLM-ийн хүлээлгийн хугацааг тооцох]:** Та bf16 форматаар 200 тэрбум параметртэй загвараас дээж авахыг хүсэж байна гэж бодъё. Энэ загвар 32 TPU v4p дээр хуваагдсан. Бүх параметрүүдийг HBM-ээс systolic array руу ачаалахад хэр хугацаа шаардагдах вэ? *Санамж: дээрх тоонуудыг ашиглаарай.*

{% details Хариуг харах бол энд дарна уу. %}

**Хариулт:** Бид 32 чип дээр `sizeof(bf16) * 200e9 = 400e9` байт ачаалж байна, энэ нь чип бүрт 12.5e9 байт гэсэн үг. Чип бүрийн HBM дамжуулах хурд 1.23e12 байна. Тиймээс ачаалах хугацаа ойролцоогоор 10 миллисекунд болно.

Энэ их дажгүй байна, учир нь *энэ нь моделээс дээж авах үед гарах хамгийн бага боломжит саатлын хугацаа* юм. Дээж авах бүрт бүх параметрүүдийг HBM-ээс ачаалах шаардлагатай тул 10 миллисекундээс бага хугацаа зарцуулах боломжгүй. Бодит амьдрал дээр, багцын хэмжээ бага үед энэ нь бараг л хүрч болохуйц байдаг.

{% enddetails %}

**Асуулт 2 [TPU дэлгэрэнгүй]:** Бүтэн TPU v5e pod-ыг авч үзье. Нийт хэдэн CPU host байдаг вэ? Хэдэн TPU TensorCore байдаг вэ? Бүх pod-ын нийт FLOPs/s хэд вэ? Нийт HBM хэд вэ? TPU v5p pod-д мөн адил тооцооллыг хий.

{% details Хариуг харахын тулд энд дарна уу. %}

**Хариулт:** TPU v5e-д, тус бүр pod нь `16x16` бөгөөд тус бүр host нь 4x2 slice байна, тэгэхээр бидэнд `16*16 / 8 = 32` host байна. TPU v5e-д, тус бүр TPU нь зөвхөн нэг core-той, тэгэхээр бидэнд 256 TensorCore байна. Нийт FLOPs/s нь bfloat16-д `16*16*2e14 = 5.1e16` байна. Тус бүр chip нь 16GB HBM-тэй, тэгэхээр энэ нь `256 * 16 = 4TB` санах ой юм.

Бүтэн TPU v5p pod-д бидэнд `16x20x28` чип байна, мөн тус бүрийн host нь 2x2x1, тэгэхээр бидэнд `16*20*28 / 2*2 = 2,240` host байна. TPU v5p-д, тус бүрийн TPU нь хоёр TensorCore-той, тэгэхээр бидэнд `8960 * 2 = 17,920` core байна. Нийт FLOPs/секунд нь bfloat16-д `8960 * 4.5e14 = 4e18` байна. Тус бүрийн чип нь 96GB HBM санах ойтой, тэгэхээр нийтдээ `8960 * 96 = 860TB` санах ой байна.

{% enddetails %}

**Асуулт 3 [PCIe ажиллагааны эрчим]:** Бид том жинтэй матриц $A$-г $\text{bfloat16}[D, F]$ төрлөөр, мөн идэвхжүүлэлтийн багц $x$-г $\text{bfloat16}[B, D]$ төрлөөр host DRAM-д хадгалах шаардлагатай гэж төсөөлье. Эдгээр дээр матриц үржүүлэх үйлдэл хийхийг хүсэж байна. Энэ нь нэг host дээр ажиллаж байгаа бөгөөд түүнд нэг TPU v6e чип холбогдсон. Та $B \ll D$, мөн $F = 4D$ гэж үзэж болно (ирээдүйн бүлгүүдэд яагаад эдгээр нь зөв таамаглал болохыг үзэх болно). FLOPs-оор PCIe-ээс хамааралгүй байхын тулд хамгийн бага багцын хэмжээ $B$ хэд байх вэ? PCIe дамжуулах хурд 1.5e10 байт / секунд гэж үзээрэй.

{% details Хариуг харахын тулд энд дарна уу. %}

**Хариулт:** Бид $2BDF$ хөвөгч цэгийн үйлдэл хийх ёстой, мөн тус бүр чип `9.2e14` хөвөгч цэгийн үйлдэл секундэд хийж чадна. Тэгэхээр энэ нь $2BDF / 9.2e14$ секунд шаардагдана. Бид DRAM-аас $2DF + 2BD$ байт ачаалах хэрэгтэй, мөн $2BF$ байтыг буцааж бичих хэрэгтэй. Бид PCIe дамжуулах хурднаар хязгаарлагдаж байгаа тул TPU руу болон TPU-гаас өгөгдөл дамжуулахад $2 \cdot (BD + DF + BF) / 1.5e10$ секунд хэрэгтэй. Бид тооцоолол нь жинг ачаалахаас удаан үргэлжлэхийг хүсэж байгаа тул, хэрвээ бид бүх жинг ачаалалтыг тооцоололтой зэрэгцүүлж чадвал, бидэнд $2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.5e10$ хэрэгтэй. Бидний таамаглалын дагуу $B \ll D$, мөн $F = 4D$ гэж үзвэл, үүнийг хялбарчилж болно.

$$\frac{8BD^2}{9.2e14} > \frac{8D^2}{1.5e10}$$

or

$$B > \frac{9.2e14}{1.5e10} \simeq 61,000$$

{% enddetails %}

**Question 4 [general matmul latency]:** Let's say we want to multiply a weight matrix int8[16384, 4096] by an activation matrix of size int8[B, 4096] where B is some unknown batch size. Let's say we're on 1 TPUv5e to start.

1. How long will this multiplication take as a function of B? *Hint: it may help to calculate how long it will take to load the arrays from HBM and how long the multiplication will actually take. Which is bottlenecking you?*
2. What if we wanted to run this operation out of VMEM? How long would it take as a function of B?

{% details Click here for the answer. %}

**Answer:** (1) The number of floating point operations we need to perform is $2 \cdot 4096 \cdot 16384 \cdot B = 1.3e8 \cdot B$. So $T_{\text{math}} = (1.3e8 \cdot B) / 3.94e14$ seconds. We need to load $16384 \cdot 4096 + 4096 \cdot B$ bytes from HBM to VMEM, and write back $16384 \cdot B$ bytes from VMEM to HBM. This means $T_{\text{comms}} = (6.7e7 + 2e4\cdot B) / 8.1e11$ seconds. Assuming as much overlap of communication and computation as possible, the whole multiplication will take approximately

$$\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{\frac{6.7e7 + 2e4\cdot B}{8.1e11}, \frac{1.3e8 \cdot B}{3.94e14}\right\}$$

We'll be FLOPs-bound when $\frac{6.7e7 + 2e4\cdot B}{8.1e11} < \frac{1.3e8 \cdot B}{3.94e14}$, or equivalently, $B > 271$. This is slightly larger than the 240 number we derive below because we factor in the full impact of $$D$$ and $$F$$.

(2) If instead we are loading from VMEM, let's consider VMEM bandwidth to the MXU as 22 times the HBM $\leftrightarrow$ VMEM bandwidth. This turns our data loading denominator from 8.1e11 to 1.78e13, and we get $B > 11$. Note that in practice, we cannot dedicate all of our VMEM bandwidth to loading $W$, so in practice it will be closer to 20.

{% enddetails %}

**Question 5 [ICI bandwidth]:** Let's say we have a TPU v5e `4x4` slice. Let's say we want to send an array of type `bfloat16[8, 128, 8192]` from `TPU{0,0}` to `TPU{3, 3}`. Let's say the per-hop latency for TPU v5e is $1\mu s$.

1. How soon will the first byte arrive at its destination?
2. How long will the total transfer take?

{% details Click here for the answer. %}

**Answer:** In a TPUv5e we have 2D connectivity. Because we have only a `4x4` slice (with no axes of size 16), we have no wraparound connections. Thus there are two ports from which our target chip can receive data, and likewise two ports from which our source chip can send data. The amount of data we have to transfer is `2 * 8 * 128 * 8192 = 1.7e7` bytes. We can transfer from both ports simultaneously (i.e. send half the array right and half down), so we get `2 * 4.5e10 = 9e10` bytes transferred per second, which means it'll take about `1.7e7 / 9e10 = 188us` to transfer the whole array through (assuming we're bandwidth bound). In a `4x4` slice, we have six hops between chips $(0, 0)$ and $(3, 3)$, since there are no wraparound links for axes with fewer than 16 chips. Since the latency of each hop is about $1\mu s$, the first byte will arrive in about`6us` and the total transfer will take `188us`.

{% enddetails %}

**Question 6 [pulling it all together, hard]:** Imagine you have a big matrix **A**: `int8[128 * 1024, 128 * 1024]` sharded evenly across a TPU v5e 4x4 slice but offloaded to host DRAM on each chip. Let's say you want to copy the entire array to TPU{0, 0} and multiply it by a vector `bf16[8, 128 * 1024]`. How long will this take? *Hint: use the numbers above.*

{% details Click here for the answer. %}

**Answer:** Let's start by outlining the operations we have to perform. Our array is about 16GB. From the table above, a TPU v5e host has a 4x2 topology, so a 4x4 has 2 hosts, Thus, since our array is evenly sharded, each host effectively contains a chunk of 1/2 of the array, or 8GB. We need to copy these chunks all to TPU{0,0}, which gives us two options:

1. We can copy over DCN and then load the entire unsharded array over PCIe into HBM.
2. We can load our sharded arrays onto their corresponding TPUs, then perform a gather over ICI, then perform the matmul on TPU{0,0}.

It should be clear that option (2) is better. DCN is slow compared to ICI and we'd much prefer to load a big array over many PCIe links rather than just a few (the 8 on host 0). Here's a diagram of part of the system. As described above, note that TPUs are connected to their neighbors by ICI (even across hosts), all TPUs are connected to their host CPU (via PCIe), and hosts are connected by DCN.

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="Each chip actually has its own PCIe link to its host, though for clarity only one is shown here." %}

Now let's work through how long each piece will take:

1. **PCIe load**: we're loading chunks of 16GB over 16 PCIe links, each of which has `1.5e10` bytes/second bandwidth. Thus this will take about 66ms.

2. **ICI copy:** each TPU now has 16GB / 16 = 1GB of our array. Our ICI bandwidth is 9e10 bytes/second per link *bidirectional*, and you'll notice from the above diagram that only 2 of the 4 ICI links on the TPU v5e are in use in this topology for TPU{0,0}. Since TPU{0,0} needs to receive a total of 15GB along 2 axes at `4.5e10` bytes/s/link, we can lower bound the time by `15e9 / (4.5e10 * 2) = 167ms`. In practice this probably isn't achievable because the load is very uneven, but it's probably within a factor of 2. As you'll see in Section 2, performing a full AllGather would also take roughly `16e9 / (4.5e10 * 2)`, so this is close to optimal.

3. **HBM $\rightarrow$ MXU load:** to perform our final matmul, we need to load these 16e9 bytes plus the bf16[8, 128 \* 1024] array (another 2MB, so negligible) over HBM bandwidth into the MXU, which will take `16e9 / 8.1e11 = 19ms`.

4. **FLOPs:** we're performing a total of $$2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7e11$$ FLOPs, мөн бид `1.97e14` bf16 FLOPs/s гүйцэтгэж чаддаг тул 1.3ms болно.

Нийт хугацааны дээд хязгаар нь эдгээр бүх хугацааны нийлбэр юм, гэхдээ TPU ихэвчлэн эдгээр үйлдлүүдийг давхар хийж чаддаг тул үүнийг pipeline-ийн асуудал гэж бодож болно. Энэ нь хамгийн удаан хэсэг дээрээ саатдаг. Хэрвээ энэ үнэн бол хариу нь ойролцоогоор 150-200мс байна.

{% enddetails %}

<h3 markdown=1 class="next-section">2-р хэсэг дууслаа! 3-р хэсэгт, хуваах болон TPU хоорондын харилцаа сэдвийг үзэх бол [энд дарна уу](../sharding).</h3>

## Нэмэлт хэсэг

### Хавсралт A: TPU-ийн дотоод бүтэцийн талаар дэлгэрэнгүй

Энд бид TPU-ийн дотоод ажиллагааг илүү гүнзгий судлах болно. Хэрэв өөрөөр заагаагүй бол бид TPU v5p-ийн техникийн үзүүлэлтийг өгөх болно.

### VPU

VPU нь TPU-гийн вектор арифметик цөм юм. VPU нь хоёр хэмжээст SIMD вектор машин (**VPU**) бөгөөд энэ нь vadd (вектор нэмэх) эсвэл vmax (элемент бүрийн хамгийн их) гэх мэт элемент бүрийн арифметик үйлдэл хийдэг. Мөн өгөгдлийг VPU болон MXU-д хадгалах зориулалттай вектор регистрүүдийг **VREGs** гэж нэрлэдэг.

**VREG-үүд:** Тус бүр TPU v5p цөмд 64 ширхэг 32-битийн VREG (TPU v4-д 32 ширхэг байсан) байдаг бөгөөд энэ нь нэг цөмд нийтдээ ойролцоогоор `64 * 8 * 128 * 4 = 256kB` хэмжээний VREG санах ойтой гэсэн үг юм (эсвэл бүх чипийн хувьд хоёр цөмтэй учраас 2 дахин их байна). TPU v5p нь VMEM-ээс нэг циклд 3 регистр ачаалж, нэг циклд 1 регистрийг VMEM рүү бичиж чадна.

**VPU:** VPU нь 2D vector arithmetic unit бөгөөд хэлбэр нь `(8, 128)` байна. Энд 128 хэмжээ нь lane тэнхлэг гэж нэрлэгддэг, 8 хэмжээ нь sublane тэнхлэг гэж нэрлэгддэг. V5 дээрх (lane, sublane) хос бүр нь 4 стандарт floating-point ALU-тэй бөгөөд эдгээр нь бие даан ажилладаг. VPU нь ихэнх arithmetic зааврыг ALU бүр дээрээ нэг циклд гүйцэтгэдэг (жишээ нь vadd буюу vector add) бөгөөд latency нь 2 цикл байдаг. Тиймээс v5 дээр та VREG-ээс f32 төрлийн 4 хос утгыг нэг циклд хамтад нь нэмэх боломжтой. VPU-ийн ердийн заавар нь `{v2 = vadd.8x128.f32 v0, v1}` гэх мэт харагддаг бөгөөд энд v0 ба v1 нь оролтын VREG-үүд, v2 нь гаралтын VREG юм.

Бүх lanes болон sublanes нь цэвэр SIMD аргаар нэг ижил програмыг цикл бүрт гүйцэтгэдэг, гэхдээ тус бүр ALU нь өөр өөр үйлдэл хийж чадна. Тиймээс бид жишээ нь нэг vadd болон нэг vsub-г нэг циклд боловсруулах боломжтой, эдгээр нь тус бүр хоёр бүтэн VREG дээр ажиллаж, гаралтыг гурав дахь VREG рүү бичдэг.

**Түргэн шалгалт [VPU дамжуулах чадварыг тооцоолох]:** Дээрх мэдээллийг ашиглан, TPU v5p хэдэн вектор FLOPs/секунд гүйцэтгэж чадахыг тооцоол. TPU v5p-ийн цагийн хурд ойролцоогоор 1.75GHz байна.

{% details Хариултыг харах бол энд дарна уу. %}

*Хариулт*: Болгоны циклд, болгоны цөм 4 векторын зааварчилгаа `8 * 128` ALU дээр гүйцэтгэж чадна. Энэ нь бүх чипийн хувьд `8 * 128 * 4 * 2` FLOPs/цикл болж байна, эсвэл `8 * 128 * 4 * 2 * 1.75e9 = 1.4e13 FLOPs/s`. Энэ нь MXU-ийн FLOPs/секундтэй харьцуулахад (ойролцоогоор `2e14`, бараг 10 дахин их) хамаагүй бага байгааг анхаарна уу.

{% enddetails %}

**Багасгалт (Reductions):** Ерөнхийдөө, дэд шугам (sublane) чиглэлээр харилцах эсвэл багасгалт хийх нь шугам (lane) чиглэлээр хийхээс илүү хялбар байдаг. Жишээ нь, VPU нь нэг шугам доторх (intra-lane) холих (shuffle) үйлдлийг дэмждэг бөгөөд энэ нь 8 хэмжээтэй тэнхлэгээр нэг циклд шилжиж чадна. Үүнийг дэд шугам чиглэлээр үр дүнтэй багасгалт хийхэд ашиглаж болно (зөвхөн 4, 2, 1-ээр холиж, 3 удаа хос элементүүдийг нийлүүлж нэмэх хэрэгтэй).

Cross-lane reduction-ууд хийх нь илүү хэцүү бөгөөд тусдаа техник хангамжийн нэгж болох XLU буюу "cross lane unit"-ийг ашигладаг. Энэ нэгж нь удаан бөгөөд нэлээд үнэтэй.

**GPU-тай харьцуулах нь:** NVIDIA GPU-дой танил хүмүүст, VPU дахь ALU бүр нь CUDA core-тэй төстэй, харин нэг VPU lane нь "Warp Scheduler"-тай төстэй, өөрөөр хэлбэл ихэвчлэн 32 CUDA Core-оос бүрдэх, SIMD тоон үйлдэл хийдэг хэсэг юм. Lane доторх reduction хийх нь амархан, гэхдээ хэрвээ бид lane хооронд мэдээлэл дамжуулах шаардлагатай бол VMEM/XLU/SMEM-ээр дамжих хэрэгтэй болдог, энэ нь их удаан байдаг. Илүү дэлгэрэнгүйг [GPU хэсэг](../gpus)-ээс үзнэ үү.

### Скалар цөм

Скаляр цөм нь TPU-ийн удирдлагын нэгж юм. Энэ нь бүх зааврыг авч, тарааж, HBM-ээс VMEM рүү өгөгдөл шилжүүлдэг бөгөөд скаляр мета өгөгдлийн ажлыг гүйцэтгэхээр програмчлагдаж болно. Скаляр цөм нь нэг урсгалтай учраас TPU-ийн тус бүр цөм нь нэг циклд зөвхөн нэг DMA хүсэлт үүсгэх боломжтой байдаг.

Энийг ойлгоход, нэг scalar цөм нь нэг VPU-г (4096 ALU-тай), 4 MXU, 2 XLU болон хэд хэдэн DMA хөдөлгүүрийг удирддаг. Нэгж тооцоолол бүрт хяналт маш их төвлөрсөн байх нь техник хангамжийн үр ашигтай байдлын эх үүсвэр болдог, гэхдээ энэ нь өгөгдлөөс хамааралтай vectorization хийх боломжийг хязгаарладаг.

### Хавсралт B: Систолик массив хэрхэн ажилладаг вэ?

TPU MXU-ийн голд `128x128` систолик массив байдаг (`256x256` нь TPU v6e дээр байна). Бүрэн ачаалалтай үед систолик массив нь 8 тактын мөчлөг тутамд нэг `bfloat16[8,128] @ bf16[128x128] -> f32[8,128]`<d-footnote>Хэрвээ та энэ тэмдэглэгээнд танил биш бол, энэ нь дараахыг хэлнэ: bfloat16 элементүүдтэй `8x128` матрицыг bfloat16 элементүүдтэй `128x128` матрицтай үржүүлж, үр дүнг float32 элементүүдтэй `8x128` матрицад хадгална.</d-footnote> үржүүлэлт хийж чадна.

* Үндсэндээ, systolic array нь 2D `128x128` (`=16,384`) сүлжээ бөгөөд тус бүр нь үржүүлэх ба нэмэх үйлдэл хийх чадвартай ALU-уудаас бүрдэнэ.
* Жингүүд (**W**, `128x128` оролт) нь дээрээс доош (RHS гэж нэрлэдэг) дамждаг бол оролтууд (**X**, `8x128` оролт) нь зүүн талаас (LHS гэж нэрлэдэг) орж ирдэг.

Энд жингүүдийн (цэнхэр) багцыг идэвхжүүлэлтүүдийн (ногоон) багцтай үржүүлэх энгийн хөдөлгөөнт зураг байна. Та жингүүд (баруун тал) эхлээд хэсэгчлэн, диагональ байдлаар ачаалагдаж байгааг анзаарна. Дараа нь идэвхжүүлэлтүүд ч бас диагональ байдлаар орж ирнэ. Доорх бүр зурагт бид давхцаж буй ногоон ба цэнхэр нэгжүүдийг бүгдийг нь үржүүлж, дээрээс ирсэн үлдэгдэлтэй нийлүүлж нэмээд, дараа нь үр дүнг нэг нэгжээр доош дамжуулна.

{% include figure.liquid path="assets/img/systolic-array.gif" %}

Энд энэ хөдөлгөөний илүү ерөнхий хувилбар байна. Энэ нь тооцооллоос гаралтын мэдээлэл хэрхэн урсаж байгааг харуулж байна:

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

Энд олон RHS болон LHS массивууд дээр энэ процессыг хэрхэн pipeline хийхийг харуулсан диаграм байна:

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

Эхэндээ pipeline-д нэг "bubble" үүсдэг, учир нь жингүүд (RHS) болон идэвхжүүлэлтүүд (LHS)-ийг ачаалж байна. Тэр эхний "bubble"-ийн дараа шинэ оролтууд болон жингүүдийг нэмэлт "bubble" үүсгэлгүйгээр ачаалж болно.

Энд bf16[2, 3] x bf16[3, 3] матрицын үржвэрийн муу анимэйшн байна. Үүнийг 2x3 жинтэй матриц болон 1 багц, 3 хэмжээтэй оролтын идэвхжүүлэлтийн matmul гэж төсөөлж болно. Энэ нь өмнөх слайдуудтай харьцуулахад эргэсэн бөгөөд оролтууд доошоо биш баруун тийш гарч байна, гэхдээ та бүтэц нь ерөнхийдөө харагдаж байна.

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

Бид энэ процессыг үр дүнтэй pipeline хийж, том матрицийг үржүүлэх боломжтой. Ингэхдээ pipeline bubble хэт том болохгүй. Гэхдээ, бидний матрицын хэмжээ MXU-ийн талын хэмжээнээс том байх ёстой. Ихэнхдээ MXU-ийн хэмжээ 128x128 байдаг. Зарим TPU (TPU v3-с эхлэн) хэд хэдэн MXU-тэй болсон, жишээ нь TPU v3-д 2 MXU, TPU v4/5-д 4 MXU байдаг. Тиймээс бид tiling-ийн хэмжээ 128 * MXU-ийн тооноос их байх хэрэгтэй. [Энд](https://www.youtube.com/watch?v=sJltBQ4MOHA) үүний талаар сайн анимейшн бий.

Trillium (TPU v6e) нь `256x256` систолик массивтай, энэ нь нэг циклд 4 дахин их FLOPs хийх боломжтой гэсэн үг. Энэ нь таны tensor-уудын хэмжээ MXU-г бүрэн ашиглахын тулд хоёр дахин том байх хэрэгтэй гэсэн үг юм.

[Энэ блог бичлэг](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu) нь тогтмол жинтэй matrix-д зориулсан systolic array үржүүлэлтийн өөр нэг маш сайн хөдөлгөөнт зургийг агуулсан байна.