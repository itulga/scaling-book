—
Энэхүү орчуулга нь MIT лицензийн дагуу эх бүтээлээс хөрвүүлэв.
Эх сурвалж: Austin et al., "How to Scale Your Model" (https://jax-ml.github.io/scaling-book/)
Орч.: Mongolian (mn)
—

---
layout: distill
title: "TPU Программыг хэрхэн профайл хийх вэ"
# permalink: /main/
description: "Энэ цуврал одоогоор онолын талаас нь авч үзсэн: техник хангамжийн roofline дээр үндэслэсэн тооцоолол. Энэ ойлголт нь их хол хүргэдэг ч ихэнх оновчлол нь практик дэлгэрэнгүй зүйлсээс хамаардаг: XLA compiler хэрхэн ажилладаг болон JAX/Tensorboard Profiler гэх мэт профайл хийх хэрэгслүүдийг хэрхэн ашиглах, тэдгээр нь ажиллахгүй үед юу хийхээ хэрхэн олох талаар. Бид үүнийг энд хэлэлцэнэ."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 9

previous_section_url: "../applied-inference"
previous_section_name: "8-р хэсэг: LLaMA-г ажиллуулах"

next_section_url: ../jax-stuff
next_section_name: "10-р хэсэг: JAX"

giscus_comments: true

authors:
  - name: Жэйкоб Остин
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
  - name: Рейнер Попе<sup>*</sup>
    url: https://x.com/reinerpope

# Өөрийн бичлэгт агуулгын жагсаалт нэмэх.
#   - TOC (Агуулгын жагсаалт) -ын нэрүүд нь тухайн хэсгийн нэртэй яг таарч байх ёстой,
#     ингэснээр бичлэг доторх холбоосууд зөв ажиллана.
#   - Доорх форматыг хэрэглэнэ үү, гараар markdown агуулгын жагсаалт бүү үүсгээрэй.
toc:
  - name: "TPU програм хангамжийн стек: Өндөр түвшний тойм"
  - name: "TensorBoard Profiler: Олон зориулалттай TPU профайлер"
  - subsections:
    - name: "Trace Viewer"
    - name: "XLA op хэрхэн унших вэ"
    - name: "Graph Viewer"
    - name: "Бодит(ойролцоо) жишээ профайл харах"
    - name: "Санах ойн профайл"
  - name: "Бодолттой бодлогууд"

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

## TPU програм хангамжийн стек-ийн ерөнхий тойм

Google нь TPU-д зориулсан олон API-г гаргаж өгдөг, өндөр түвшний JAX кодоос эхлээд доод түвшний Pallas эсвэл HLO хүртэл. Ихэнх программистууд зөвхөн JAX код бичдэг. Энэ нь танд NumPy-тай төстэй, шугаман алгебрийн програмыг 추상 байдлаар бичих боломж олгодог бөгөөд эдгээр програмыг автоматаар компайл хийж, TPU дээр үр дүнтэй ажиллуулдаг.

Энд энгийн жишээ байна, хоёр матрицыг хооронд нь үржүүлдэг JAX програм:

```py
import jax
import jax.numpy as jnp

def multiply(x, y):
  return jnp.einsum('bf,fd->db', x, y)

y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
```

`jax.jit`-ийг дуудахад, бид JAX-д энэ функцийг мөрдөж, ML тооцоололд зориулсан платформоос хамааралгүй IR болох [StableHLO](https://openxla.org/stablehlo) нэртэй доод түвшний IR-ийг гаргахыг хэлдэг. Энэ IR-ийг дараа нь XLA compiler нь HLO болгон бууруулдаг. Compiler нь олон удаагийн дамжуулалт хийж, нэгтгэл, байрлал, болон бусад хүчин зүйлсийг тодорхойлдог бөгөөд энэ нь JAX профайл дээр харагдах HLO-г үүсгэдэг. Энэ HLO нь JAX код дахь бүх үндсэн шугаман алгебрийн үйлдлүүдийг (matmuls, pointwise ops, convolutions гэх мэт) LLVM-стайл граф хэлбэрээр илэрхийлдэг. Жишээ нь, дээрх програмын товч хувилбар нь HLO хэлбэрээр дараах байдалтай байна<d-footnote>Энэ HLO-г авахын тулд та `jax.jit(f).lower(*args, **kwargs).compile().as_text()`-ийг ажиллуулж болно.</d-footnote>:

```c
ENTRY %main.5 (Arg_0.1: f32[128,256], Arg_1.2: bf16[256,16]) -> f32[16,128] {
  %Arg_1.2 = bf16[256,16]{1,0} parameter(1), metadata={op_name="y"}
  %convert.3 = f32[256,16]{1,0} convert(bf16[256,16]{1,0} %Arg_1.2),
  %Arg_0.1 = f32[128,256]{1,0} parameter(0), metadata={op_name="x"}
  ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1},
}
```

Бид HLO-ийн синтаксийг удахгүй тайлбарлах болно, гэхдээ одоохондоо дээрх JAX кодтой их төстэй гэдгийг анзаараарай. Жишээ нь,

```c
ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
```

энэ бол дээрх бодит matmul бөгөөд энэ нь хоёр f32 матрицын 0 ба 1 хэмжээст дагуу үржүүлдэг.

**Энэ HLO-г TPU дээр ажиллуулах боломжтой код болгон хувиргахын тулд XLA compiler эхлээд үүнийг LLO** (low-level optimizer) IR болгон бууруулдаг. LLO нь TPU-г шууд удирддаг, санах ойн хооронд хуулбар хийхийг зохион байгуулдаг, массивуудыг systolic array руу илгээдэг гэх мэт. LLO код нь буферүүдийг systolic array руу илгээх, үр дүнг авах, TPU санах ойн өөр өөр хэсгүүдийн хооронд мэдээлэл дамжуулах DMA-г зохион байгуулах үндсэн үйлдлүүдийг агуулдаг. Энэ кодыг LLO болгож бууруулсны дараа машин код руу хөрвүүлж, TPU-ийн IMEM-д ачаалж, ажиллуулдаг.

Хэрвээ програм маань бидний хүссэнээс удаан ажиллаж байвал, бид голчлон JAX түвшинд ажиллаж, гүйцэтгэлийг сайжруулдаг. Гэхдээ ингэхийн тулд бид HLO-ийн утга санаа болон код хэрхэн TPU дээр ажиллаж байгааг ойлгох хэрэгтэй болдог. Хэрвээ доод түвшинд ямар нэгэн асуудал гарвал, бид өөр нэг арга хэрэглэж, [Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html) дээр өөрийн гэсэн kernel бичдэг. Програмын HLO болон түүний гүйцэтгэлийн статистикийг харахын тулд бид JAX profiler ашигладаг.

## JAX Profiler: Олон Үйлдэлт TPU Profiler

JAX нь олон зориулалттай TPU profiler-ийг санал болгодог бөгөөд энэ нь програм ажиллах үед TPU дээр юу болж байгааг ойлгоход хэрэгтэй олон хэрэгсэлтэй. Та `jax.profiler` модулийг ашиглан програмыг ажиллаж байх үед нь trace хийж, тус бүрийн дэд хэсгийн үргэлжлэх хугацаа, програм бүрийн HLO, санах ойн ашиглалт болон бусад мэдээллийг бичиж авч болно. Жишээ нь, энэ код нь trace-ийг `/tmp/tensorboard` доторх файл руу хадгална. Үүнийг TensorBoard дээр харах боломжтой ([энд](https://docs.jax.dev/en/latest/profiling.html#tensorboard-profiling) алхам алхмаар заавар байна).

```python
import jax
with jax.profiler.trace("/tmp/tensorboard"):
  key = jax.random.key(0)
  x = jax.random.normal(key, (1024, 1024))
  y = x @ x
  y.block_until_ready()

# Now you can load TensorBoard in a Google Colab with
#
# !pip install tensorboard tensorboard-plugin-profile
# %load_ext tensorboard
# %tensorboard --logdir=/tmp/tensorboard
#
# or externally with
#
# > tensorboard --logdir=/tmp/tensorboard
#
```

Энд профайлэрт юу хийж болох талаар товчхон танилцуулга байна:

{% include figure.liquid path="assets/img/xprof-overview.png" class="img-fluid" %}

TensorBoard-д орсны дараа, profiler нь таны програмыг ойлгоход туслах хэдэн гол табтай:

1. **Trace Viewer** нь TPU дээр яг юу болж байгааг дэлгэрэнгүй цагийн шугамаар харуулна.
2. **Graph Viewer** нь HLO графыг харуулдаг бөгөөд энэ нь програмын аль хэсгүүд хоорондоо холбогдож, хэрхэн хэсэглэгдэж байгааг харуулна.
3. **Memory Profile болон Memory Viewer:** эдгээр нь таны програм хэр их санах ой ашиглаж байгааг харуулна.

Профайл хуваалцах нь бага зэрэг хэцүү байдаг ч, [энд](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a) энгийн Transformer-ийн Trace Viewer бүрэлдэхүүн хэсэгтэй Perfetto холбоос байна. [Энэ Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) нь танд бүтэн JAX/TensorBoard trace үүсгэх болон түүн дээр туршиж үзэх боломжийг олгоно.

### Мөр хянагч

**Trace Viewer нь профайлерын хамгийн хэрэгтэй хэсэг байж магадгүй.** Доорх жишээ нь энгийн Transformer-ийг хэсгүүдээр нь тайлбарласан байна. Нэрүүд нь кодонд өгсөн шошгоноос авсан.

{% include figure.liquid path="assets/img/trace-viewer.png" class="img-fluid" %}

Trace Viewer нь бүх TPU цөм дээрх үйлдлүүдийн цаг хугацааны дарааллыг харуулдаг. Бид энд зөвхөн TPU:0-ийг харж байна, учир нь ихэнхдээ бүх TPU-ууд ижил заавар гүйцэтгэдэг. Хэдэн чухал тэмдэглэлүүд:

1. Дээд мөр (XLA Ops) нь бодит TPU үйлдлүүдийг харуулж байна (нэрс нь HLO нэрс юм). Бусад бүх зүйл нь `jax.named_scope`, `jax.named_call`, болон Python stack trace дээр үндэслэсэн ойролцоо мөрдлөг юм.
2. Давтагдсан блокуудыг анзаарахад, бид энд нэг давхаргыг тусгаарлаж чадна. Мөн (кодыг хараад/Transformer хэрхэн ажилладгийг ойлгоод) аль хэсэг нь attention, аль хэсэг нь MLP болохыг харж болно.
3. XLA үйлдэл дээр дарснаар, энэ нь кодын аль хэсгээс гарч ирсэн болохыг харж болно (мөрдлөгийг ойлгоход хэрэгтэй) мөн Graph viewer рүү холбоосыг харж болно.

<p markdown=1 class="takeaway">**Зөвлөгөө:** Та Trace Viewer-ыг "видео тоглоом" шиг удирдаж болно. A/D товчоор зүүн, баруун тийш хөдлөх, W/S товчоор ойртуулах, холдуулах боломжтой. Эдгээр удирдлагаар хайлт хийх нь илүү хялбар болдог.</p>

### XLA op-ийг хэрхэн унших вэ

HLO-г унших нь үнэндээ тийм ч хэцүү биш, мөн дээрх trace-ийн аль хэсэгтэй таарч байгааг ойлгоход маш их тусалдаг. Энд fusion.3 гэж нэртэй жишээ op байна.

```py
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

Энийг хэсгүүдэд нь хувааж үзье.

* **Оп нэр**: fusion.3
  * Dot эсвэл fusion оп гэдэг нь хамгийн ихдээ 1 matrix multiplication болон холбоотой pointwise VPU-ops-уудыг агуулсан үйлдлүүдийн багц юм.
* **Хэлбэр/байршил**: `bf16[32,32,4096]`
  * Энэ нь оп-ийн гаралтын хэлбэр юм. Бид dtype нь bf16 (1 параметрт 2 байт) бөгөөд `[32,32,4096]` нь хэлбэр гэдгийг харж байна.
* **Байршил:** `{2,1,0:T(8,128)(2,1)}`
  * `{2,1,0:T(8,128)(2,1)}` нь санах ойд тэнхлэгүүдийн дарааллыг (багана гол, мөр гол гэх мэт) болон array padding-ийг заадаг. Доор дэлгэрэнгүй.
* **Санах ойн байрлал:** S(1)
  * S(1) нь энэ array нь VMEM-д байгаа гэсэн үг. S(0) (заримдаа бичигдэхгүй) нь HBM. S(2) ба S(3) нь өөр санах ойн орон зайнууд.
* **Параметрүүд**: `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32`
  * Энэ оп нь нэг оролттой, fusion.32 гэж нэрлэгдсэн, тодорхой хэлбэртэй bf16 array юм. Энэ нь ямар function энэ рүү өгөгдөл дамжуулж байгааг харуулна.

Энэ тэмдэглэлийг арай илүү ойлгож үзье. Үүнийг энгийн жишээ болгон авч үзье:

`f32[3,5]{1,0:T(2,2)}`

энэ нь дахин бидэнд энэ Op нь float32 төрлийн массивыг `[3, 5]` хэлбэртэйгээр буцаадаг бөгөөд тодорхой tiling-тай `{1,0:T(2,2)}` гэдгийг хэлж байна. Tilings нь *тийм ч* чухал биш ч, товчхон хэлэхэд, tilings нь N- хэмжээст массив санах ойд хэрхэн дарааллаар байрлаж байгааг хэлж өгдөг. Энэ массив хэрхэн байрлаж байгааг харуулсан зураг энд байна:

{% include figure.liquid path="assets/img/tiling.png" class="img-fluid" %}

`{1,0:T(2,2)}`-дотор, `1,0` хэсэг нь массивын хэмжээсүүдийн дарааллыг физик санах ойд хэрхэн байрлаж байгааг, хамгийн багаас хамгийн их хүртэл заадаг. Энэ хэсгийг баруунаас зүүн тийш уншаад, `f32[3,5]`-д байгаа хэмжээсүүдийг сонгож, массивын физик байрлал ямар байгааг олж мэдэж болно. Энэ жишээнд, физик байрлал нь `[3,5]`, логик хэлбэртэй ижил байна.

Үүний дараа, `T(2,2)` нь массивыг `(2, 2)`-ийн хэмжээтэй хэсгүүдэд хуваасан болохыг харуулна. Хэсэг бүрийн дотор массив эхлээд мөрөөр (**row-major**), дараа нь баганаар, өөрөөр хэлбэл `(0, 0)`-ыг дагаад `(0, 1)`, дараа нь `(1, 0)` болон `(1,1)` байна. `T(2, 2)`-ийн хуваалттай учраас массивыг `[4, 6]` хүртэл дүүргэж, санах ойн хэрэглээг ойролцоогоор 1.6 дахин ихэсгэдэг. Дээр өгсөн том bf16 массивын хувьд, `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}`, бид `T(8,128)(2,1)` хийдэг бөгөөд энэ нь массив хоёр түвшний хуваалттай болохыг харуулна: гадна `(8, 128)` хуваалт болон дотор `(2, 1)` хуваалт (bf16-д ашиглагддаг тул ачаалал үргэлж 4 байтын үржвэр байх ёстой). Жишээ нь, энд `bf16[4,8]{1,0,T(2,4)(2,1)}` байна (өнгө нь (2,4) хуваалт, улаан хүрээ нь (2,1) хуваалт):

{% include figure.liquid path="assets/img/tiling2.png" class="img-fluid img-small" %}

Тайл хийх нь тэнзоруудын хэсгүүдийг VMEM-д хэр үр дүнтэй ачаалахыг нөлөөлж болно. Заримдаа XLA нь тэнзорыг програм дотор "дахин тайл" эсвэл "дахин байрлуулах" хуулбаруудыг үүсгэдэг бөгөөд энэ нь заримдаа ихээхэн нэмэлт зардалтай байдаг.<d-footnote>JAX нь энэ асуудлыг тойрч гарах туршилтын боломжийг санал болгодог. Энэ нь XLA-д програмын оролтын "илүүд үздэг" байрлалыг тооцоолох боломжийг олгодог. Та програмыг `jax.jit` ашиглан "just-in-time" компайл хийх үедээ ихэвчлэн JAX-д ямар хэлбэр (shape), өгөгдлийн төрөл (dtype) хүлээж авахыг хэлдэг "mock" оролтуудыг дамжуулдаг. Эдгээр нь ихэвчлэн төгс биш тайл хийх мэдээлэл агуулж байдаг. Үүний оронд та оролтын байрлалыг AUTO гэж зааж өгч болно, тэгвэл `jax.jit` нь jitted програмд хамгийн тохиромжтой байрлалыг буцаана. Та дараа нь тэр байрлалаар тэнзорыг шууд ачаалж, програм дотор илүү хуулбар үүсэхээс сэргийлж чадна.</d-footnote>

### Граф Үзэгч

Дээрх зарим нэгтгэлүүд төвөгтэй санагдаж болох ч, XLA Graph Viewer тэднийг ойлгоход хялбар болгодог. Жишээ нь, энд нэлээн төвөгтэй нэгтгэлийн харагдац байна:

{% include figure.liquid path="assets/img/graph-viewer.png" class="img-fluid" %}

Олон HLO графыг хараад, HLO үйлдлүүдийг та профайл хийж байгаа код дээрээ тааруулахыг оролдох нь үнэхээр хэрэгтэй. Хайрцган дээр хулганаа аваачвал ихэнхдээ тухайн функц тодорхойлогдсон кодын мөрийг харуулдаг.

### Жинхэнэ (ойролцоогоор) жишээ профайл харах

[Энэ Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) нь хуурамч Transformer-ийн жишээ profile-ыг агуулсан. [Энд](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a) Perfetto холбоос байна. Та яаралтай бол Trace Viewer-ийг дор хаяж харах боломжтой. Би ердийнхөөсөө илүү хичээж trace дээр `jax.named_scope` дуудлагуудыг тайлбарласан тул юу болж байгааг хялбархан олж харж болно.

{% include figure.liquid path="assets/img/transformer-xprof.png" class="img-fluid" %}

Профайлыг харж, хэсэг бүр юу хийж байгааг сайн ойлгохыг хичээгээрэй. Үүнийг жаахан задлая, эхлээд FFW блокоос эхэлье:

{% include figure.liquid path="assets/img/transformer-ffw.png" class="img-fluid" %}

Энд бид FFW блок руу ойртож харлаа. Та up-projection Op нь нэгтгэл (matmul) бөгөөд оролтууд нь `bf16[8, 1024, 8192]` болон `bf16[8192, 16384]`, гаралт нь `bf16[32, 1024, 16384]` гэдгийг харна. Би мэдэж байна (учир нь энэ кодыг би бичсэн) энэ нь 4-way DP, 2-way MP хуваагдсан matmul-ийн локал харагдац юм, тиймээс бид үнэндээ хийж байгаа зүйл бол

**X:** `bf16[32, 1024, 8192]` \* **W<sub>in</sub>**: `bf16[8192, 32768]` -> **Tmp**: `bf16[32, 1024, 32768]`

**Бид үүнийг хэр удаан үргэлжлэх гэж бодож байна вэ?** Юуны өмнө, манай data parallel shard бүрийн batch size нь `8 * 1024 = 8192`, тиймээс бид ихэнхдээ тооцооллын хязгаартай байна. Энэ нь 8 TPUv2 core дээр (Google Colab дээр үнэгүй ашиглах боломжтой), тиймээс бид үүнийг ойролцоогоор `2 * 32 * 1024 * 8192 * 32768 / (23e12 * 8) = 95.6ms` хугацаанд үргэлжлэх байх гэж бодож байна. Энэ нь яг үнэндээ 96ms орчим үргэлжилдэг. Энэ бол маш сайн! Энэ нь бид FLOPs-ыг маш сайн ашиглаж байна гэсэн үг!

**Харилцаа холбоо ямар вэ?** Та хоёр дахь matmul-ийн төгсгөлд жижиг fusion нуусан байгааг анзаарах болно. Хэрвээ бид үүн дээр дарвал, та харах болно

```py
%fusion.1 = bf16[8,1024,4096]{2,1,0:T(8,128)(2,1)} fusion(bf16[8,1024,8192]{2,1,0:T(8,128)(2,1)} %fusion.31), kind=kCustom, calls=%all-reduce-scatter.1
```

энэ нь үндсэндээ жижиг ReduceScatter юм (энэ бол GraphViewer);

{% include figure.liquid path="assets/img/reduce-scatter-xprof.png" class="img-fluid" %}

Энэ хэр удаан үргэлжлэх вэ гэж бид хүлээж байна вэ? Бид TPUv2 4x2 дээр ReduceScatter хийж байна, энэ нь зөвхөн нэг hop шаардана, 1.2e11 хоёр чиглэлийн bandwidth ашиглана. Массивын хэмжээ `2*32*1024*8192`, batch тэнхлэгийг 4 хэсэгт хуваасан, тэгэхээр нэг shard нь `2*8*1024*8192=134MB` байна. Тэгэхээр энэ нь ойролцоогоор 1.1ms үргэлжлэх ёстой. **Бодитоороо хэр удаан үргэлжилдэг вэ?** Профайл дээр 1.13ms гэж гарсан. Тэгэхээр бид roofline-д маш ойрхон байна!

**Анхаарлыг бас харцгаая!** Энэ бол анхаарал (attention) бүрэлдэхүүний танилцуулга:

{% include figure.liquid path="assets/img/attn-xprof.png" class="img-fluid" %}

Би Q projection op дээр дарсан, энэ нь $$W_Q$$ of shape [d<sub>model</sub> = 8192, n<sub>heads</sub> = 32, d<sub>qkv</sub> = 256]. We're Megatron sharding along the head dimension. Try to do the same exercise of calculating how long these should take.

### Memory Profile

The Memory Profile makes it easy to see the program memory as a function of time. This is helpful for debugging OOMs. You can see here about 7.5GB allocated to model parameters and about 10GB free. So we can fit a lot more into memory.

{% include figure.liquid path="assets/img/memory-viewer.png" class="img-fluid" %}

## Worked Problems

**Question 1**: take a look at [this](https://colab.research.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli?usp=sharing) Colab/profile and figure out what looks suspicious and what's going on here. Can you tell me exactly what computations are happening and what each operation is doing? What are the true shapes of each matrix involved and how are they sharded? *Try looking at the profile first without reading the code.*

{% include figure.liquid path="assets/img/all-reduce-profile.png" class="img-fluid" %}

{% details Click here for the answer. %}

This is two matrix multiplications, i.e. specifically this:

```py
def matmul(w1, w2, x):
  return jnp.einsum('wf,bf->bw', w2, jnp.einsum('fw,bw->bf', w1, x))
```

You can see a reduce, two big fusions, and an all-reduce. The first big fusion is:

```%fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1```

which tells us the per-shard shape is `bf16[8192] * bf16[4096, 8192] -> bf16[4096]` (over the 8192 dimension). By observing the final AllReduce with `replica_groups=\{\{0,16,32,48,64,80,96,112\}, ...\}`, we can tell we're doing 8-way model parallelism, so the true shapes are `[8, 8192] * bf16[32,768, 8192] -> bf16[8, 32,768]`.

{% enddetails %}

**Question 2:** [The Transformer Colab from earlier](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) implements a simple mock Transformer. Follow the instructions in the Colab and get a benchmark of the naive Transformer with GSPMD partitioning. How long does each part take? How long should it take? What sharding is being used. Try fixing the sharding! *Hint: use `jax.lax.with_sharding_constraints` to constrain the behavior. With this fix, what's the best MXU you can get?*

For reference, the initial version gets roughly 184ms / layer and the optimized profile gets 67 ms / layer. Once you've done this, try staring at the profile and see if you can answer these questions purely from the profile:

- What sharding strategy is this?
- What is the batch size, $$d_\text{model}$$, $$d_\text{ff}$$ матриц ашигладаг уу?
- Анхаарал (attention) болон MLP блок дээр цагийн ямар хэсэг зарцуулагддаг вэ?
- Roofline дээр тус бүрийн op-д ямар хэмжээний цаг зарцуулах ёстой вэ?

**Тэмдэглэл:** Энэ асуудлыг бичсэнээс хойш XLA compiler илүү сайн болсон. Эхний хувилбар нь одоо ойролцоогоор 90мс / давхарга болсон ба сайжруулсан хувилбар нь ердөө 10мс / давхаргаар илүү хурдан (80мс / давхарга) байна. Гэсэн ч, туршиж үзэхэд сонирхолтой бөгөөд та үүнээс илүү сайн хийж чадах эсэхээ хараарай.

<h3 markdown=1 class="next-section">9-р хэсэг дууслаа. 10-р хэсэгт JAX-ийн зэрэгцээ ажиллагааны талаар дэлгэрэнгүй үзэх бол [энд](../jax-stuff) дарна уу.</h3>
