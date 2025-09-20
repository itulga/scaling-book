—
Энэхүү орчуулга нь MIT лицензийн дагуу эх бүтээлээс хөрвүүлэв.
Эх сурвалж: Austin et al., "How to Scale Your Model" (https://jax-ml.github.io/scaling-book/)
Орч.: Mongolian (mn)
—

---
layout: distill
title: "GPU-г хэрхэн ойлгох вэ"
description: "Бид Google-д TPU-д дуртай, гэхдээ GPU ч бас маш сайн. Энэ бүлэгт GPU-гийн ертөнц рүү гүнзгий орж үзнэ – чип бүр хэрхэн ажилладаг, тэдгээрийг хэрхэн сүлжээнд холбодог, энэ нь LLM-д ямар утгатай вэ, ялангуяа TPU-тай харьцуулахад. NVIDIA, AMD, Intel болон бусад олон төрлийн GPU архитектур байдаг ч энд бид зөвхөн NVIDIA GPU-д төвлөрнө. Энэ хэсэг нь <a href='https://jax-ml.github.io/scaling-book/tpus/'>2-р бүлэг</a> болон <a href='https://jax-ml.github.io/scaling-book/training'>5-р бүлэг</a>-д тулгуурласан тул эхлээд тэдгээрийг уншихыг зөвлөж байна."
date: 2025-08-18
future: true
htmlwidgets: true
hidden: false

хэсгийн_дугаар: 12

previous_section_url: "../conclusion"
previous_section_name: "11-р хэсэг: Дүгнэлт"

next_section_url:
next_section_name: "Төгсгөл"

ном зүй: main.bib

giscus_comments: үнэн

authors:
  - name: Жэйкоб Остин<sup>†</sup>
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: <sup>†</sup>Google DeepMind
  - name: Свапнил Патил<sup>†</sup>
    url: "https://www.linkedin.com/in/swapnil-patil-5b47a068"
  - name: Адам Паскэ<sup>†</sup>
    url: https://x.com/apaszke
  - name: Рейнер Попе<sup>*</sup>
    url: https://x.com/reinerpope
    affiliations:
      name: <sup>*</sup>MatX

toc:
  - name: GPU гэж юу вэ?
  - subsections:
    - name: Санах ой
    - name: "GPU-ийн техникийн үзүүлэлтийн хураангуй"
    - name: GPU ба TPU-ийн ялгаа (чип түвшинд)
    - name: "Асуулт 1: GPU техник хангамж"
  - name: Сүлжээ (Networking)
  - subsections:
    - name: Нэг нод (node) дээр
    - name: "Асуулт 2: GPU-нодууд"
    - name: Нодын түвшнээс цааш
    - name: "Асуулт 3: Нодын түвшнээс цааш"
  - name: GPU дээр Collective-ууд хэрхэн ажилладаг вэ?
  - subsections:
    - name: Нэг нод доторх collective-ууд
    - name: Нодууд хоорондын collective-ууд
    - name: "Асуулт 4: Collective-ууд"
  - name: "GPU дээр LLM масштаблах Roofline-ууд"
  - subsections:
    - name: "Өгөгдлийн параллелизм"
    - name: "Тэнцэр (Tensor) параллелизм"
    - name: "Мэргэжилтний (Expert) параллелизм"
    - name: "Дамжуулах (Pipeline) параллелизм"
    - name: "Жишээнүүд"
    - name: "GPU дээр LLM масштаблахын товч агуулга"
    - name: "Асуулт 5: LLM roofline-ууд"
  - name: "Талархал ба Дэлгэрэнгүй унших материал"
  - name: "Хавсралт"
  - subsections:
    - name: "Хавсралт A: GB200-той бол юу өөрчлөгдөх вэ?"
    - name: "Хавсралт B: Сүлжээний дэлгэрэнгүй"

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
---

## GPU гэж юу вэ?

Орчин үеийн ML GPU (жишээ нь H100, B200) нь үндсэндээ матриц үржүүлэхэд (Matrix Multiplication) зориулсан олон тооны тооцооллын цөмүүдээс (эдгээрийг **Streaming Multiprocessors** буюу **SMs** гэж нэрлэдэг) бүрддэг бөгөөд эдгээр нь хурдан санах ойтой (үүнийг **HBM** гэж нэрлэдэг) холбогдсон байдаг. Энд нэг зураг байна:

{% include figure.liquid path="assets/gpu/gpu-diagram.png" class="img-fluid" link="true" caption="<b>Зураг:</b> H100 эсвэл B200 GPU-ийн ерөнхий бүтэцтэй диаграмм. H100 нь 132 SM-тэй, харин B200 нь 148 SM-тэй. Бид 'Warp Scheduler' гэдэг нэр томьёог өргөн утгаар нь хэрэглэж байгаа бөгөөд энэ нь 32 CUDA SIMD цөм <i>ба</i> тэдэнд ажил хуваарилдаг scheduler-г хэлж байна. Энэ нь TPU-тэй хэр адилхан харагдаж байгааг анхаарна уу!" %}

Тус бүр SM нь TPU-ийн Tensor Core шиг зориулалтын матриц үржүүлэх цөмтэй (харамсалтай нь мөн **Tensor Core** гэж нэрлэдэг<d-footnote>GPU Tensor Core нь SM-ийн матриц үржүүлэх дэд нэгж бөгөөд TPU TensorCore нь MXU, VPU болон бусад бүрэлдэхүүн хэсгүүдийг агуулсан нэгж юм.</d-footnote>), вектор тооцооллын нэгжтэй (энэ нь **Warp Scheduler** гэж нэрлэгддэг<d-footnote>NVIDIA-д үүнийг сайн нэрлэсэн нэр байхгүй тул бид хамгийн боломжит муу сонголтоор нь хэрэглэж байна. Warp Scheduler нь голчлон ажил үүргийг хэд хэдэн CUDA цөмд хуваарилагч нэгж боловч бид энд удирдлагын нэгж болон түүний хянадаг цөмүүдийн багцыг илэрхийлэхэд хэрэглэж байна.</d-footnote>), мөн хурдан дотоод санах ойтой (энэ нь **SMEM** гэж нэрлэгддэг). TPU-ээс ялгаатай нь, хамгийн ихдээ 2 тусдаа "Tensor Core"-той байдаг бол, орчин үеийн GPU нь 100-аас дээш SM-тай (H100 дээр 132 ширхэг). Эдгээр SM тус бүр нь TPU-ийн Tensor Core-оос хүч багатай ч систем бүхэлдээ илүү уян хатан байдаг. SM бүр бараг бүрэн бие даасан тул GPU нь нэгэн зэрэг хэдэн зуун тусдаа үүрэг гүйцэтгэж чадна.<d-footnote>SM-ууд бие даасан боловч, ихэнхдээ хамгийн өндөр гүйцэтгэлд хүрэхийн тулд хамтран ажиллах шаардлагатай байдаг. Учир нь тэд бүгд багтаамж хязгаартай L2 кэш санах ойг хуваалцдаг.</d-footnote>

H100 SM-ийн илүү дэлгэрэнгүйг харцгаая:

{% include figure.liquid path="assets/gpu/blackwell-sm.png" class="img-small" link="true" caption="<b>Зураг:</b> H100 SM-ийн диаграмм (<a href='https://wccftech.com/nvidia-hopper-gh100-gpu-official-5nm-process-worlds-fastest-hpc-chip-80-billion-transistors-hbm3-memory/'>эх сурвалж</a>). Энэ нь 4 <i>дэд хэсэг</i>-ийг харуулж байна. Тус бүр нь Tensor Core, Warp Scheduler, Register File болон өөр өөр нарийвчлалтай CUDA Cores-уудтай. Доор байрлах 'L1 Data Cache' нь 256kB SMEM нэгж юм. B200 нь үүнтэй төстэй харагддаг, гэхдээ Tensor Core-уудыг хангахын тулд илүү их Tensor Memory (TMEM)-ийг нэмсэн." %}

Тус бүр SM нь 4 ижил хэсэгт хуваагддаг бөгөөд NVIDIA эдгээрийг **SM дэд хэсэг** гэж нэрлэдэг. Дэд хэсэг бүрт нэг Tensor Core, 16к 32-бит бүртгэгч, мөн SIMD/SIMT вектор тооцооллын нэгж буюу Warp Scheduler байдаг. Энэ нэгжийн шугамуудыг (ALU) NVIDIA нь **CUDA Cores** гэж нэрлэдэг. Дэд хэсэг бүрийн гол бүрэлдэхүүн хэсэг нь Tensor Core бөгөөд энэ нь матриц үржвэрийг гүйцэтгэдэг ба FLOPs/s-ийн ихэнх хувийг бүрдүүлдэг. Гэхдээ энэ нь цорын ганц чухал бүрэлдэхүүн хэсэг биш юм.

* **CUDA цөмүүд:** тус бүрийн дэд хэсэг нь ALU-уудын багц болох CUDA цөмүүдтэй бөгөөд эдгээр нь SIMD/SIMT вектор арифметик үйлдэл хийдэг. ALU бүр ихэвчлэн нэг циклд нэг арифметик үйлдэл хийж чадна, жишээ нь f32.add.<d-footnote>Шинэ GPU-ууд FMA (Fused-Multiply Add) зааврыг дэмждэг бөгөөд энэ нь нэг циклд хоёр FLOP хийдэг. NVIDIA энэ боломжийг ашиглан үзүүлэлтээ хоёр дахин ихээр зарладаг.</d-footnote> Дэд хэсэг бүр 32 fp32 цөмтэй (мөн цөөн тооны int32 ба fp64 цөмтэй) бөгөөд бүгд нэг циклд ижил заавар гүйцэтгэдэг. TPU-ийн VPU шиг, CUDA цөмүүд нь ReLU, pointwise вектор үйлдэл, мөн бууруулалт (нийлбэр) хийх үүрэгтэй.<d-footnote>Түүхийн хувьд, Tensor Core-оос өмнө CUDA цөмүүд нь GPU-ийн гол бүрэлдэхүүн хэсэг байсан бөгөөд дүрслэл хийхэд ашиглагддаг байсан, үүнд цацраг-гурав өнцөгт огтлолцол болон сүүдэрлэлт орно. Өнөөгийн тоглоомын GPU-ууд дээр эдгээр цөмүүд нь ихэнх дүрслэлийн ажлыг хийдэг хэвээр байна, харин TensorCore-ууд нь up-sampling (DLSS) хийхэд ашиглагддаг. Энэ нь GPU-д бага нягтаршилтайгаар дүрслэл хийх (бага пиксел = бага ажил) боломж олгож, ML ашиглан дүрслэлийг томруулдаг.</d-footnote>

* **Tensor Core (TC):** тус бүрийн дэд хэсэг өөрийн Tensor Core-тэй байдаг, энэ нь матриц үржүүлэх зориулалттай тусгай нэгж бөгөөд TPU MXU шиг ажилладаг. Tensor Core нь GPU-ийн FLOPs/s-ийн ихэнх хувийг эзэлдэг (жишээ нь H100 дээр 990 bf16 TC TFLOP/s байдаг бол зөвхөн CUDA core-оос 66 TFLOPs/s байдаг).
  * [990 bf16 TFLOPs/s](https://www.nvidia.com/en-us/data-center/h100/) нь 132 SM 1.76GHz дээр ажиллахад, нэг H100 TC нь `7.5e12 / 1.76e9 / 4 ~ 1024` bf16 FLOPs/cycle хийж чадна, энэ нь ойролцоогоор 8x8x8 матриц үржүүлэлт юм.<d-footnote>NVIDIA компани TC-ийн техник хангамжийн нарийн мэдээллийг ихэвчлэн хуваалцдаггүй тул энэ нь зөвхөн таамаглал бөгөөд баттай үнэн биш – яг TC хэрхэн хийгдсэн тухай биш юм. Бид мэдэж байгаа нь V100 нь 256 FLOPs/TC/cycle хийж чаддаг. A100 нь 512, H100 нь 1024, B200-ийн мэдээлэл албан ёсоор гараагүй ч магадгүй 2048 FLOPs/TC/cycle байх боломжтой, учир нь `2250e12 / (148 * 4 * 1.86e9)` нь 2048 орчим байна. Илүү дэлгэрэнгүй мэдээллийг <a href='https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727'>энд</a> харж болно.</d-footnote>
  * TPU шиг, GPU нь бага нарийвчлалтай матриц үржүүлэлтийг илүү хурдан хийж чадна (жишээ нь H100 нь fp8 FLOPs/s нь fp16-гаас 2 дахин их). Бага нарийвчлалтай сургалт эсвэл үйлчилгээ илүү хурдан байж болно.
  * Volta-гаас хойших бүх GPU үе бүр өмнөх үеэсээ TC-ийн хэмжээг нэмэгдүүлсэн ([энэ талаар сайн нийтлэл](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)). B200 дээр TC маш том болсон тул оролтууд нь SMEM-д багтахаа больсон, тиймээс B200 нь TMEM гэдэг шинэ санах ойн орон зайг нэвтрүүлсэн.<d-footnote>Ampere дээр Tensor Core-г нэг warp-оос өгч болдог байсан бол Hopper дээр бүтэн SM (warpgroup) хэрэгтэй болсон, харин Blackwell дээр 2 SM-ээс өгдөг болсон. Blackwell дээр матриц үржүүлэлтийн хэмжээ маш том болсон тул аргументууд (ялангуяа accumulator) нь register memory/SMEM-д багтахаа больсон, тиймээс Blackwell нь TMEM-г нэмсэн.</d-footnote>

**CUDA цөмүүд TPU-ийн VPU-ээс илүү уян хатан:** GPU-ийн CUDA цөмүүд (V100-с хойш) SIMT (*Single Instruction Multiple Threads*) программчлалын загвар ашигладаг, харин TPU нь SIMD (*Single Instruction Multiple Data*) загвар ашигладаг. TPU-ийн VPU дахь ALU-ууд шиг, нэг subpartition доторх бүх CUDA цөмүүд нэгэн зэрэг ижил үйлдэл хийх ёстой (жишээ нь, хэрвээ нэг цөм хоёр float нэмэж байвал, тухайн subpartition доторх бусад бүх CUDA цөмүүд бас нэмэх үйлдэл хийх ёстой). Гэхдээ VPU-ээс ялгаатай нь, тус бүрийн CUDA цөм (эсвэл CUDA программчлалын загварт "thread" гэж нэрлэдэг) өөрийн гэсэн instruction pointer-тай бөгөөд тусдаа _программчлагдах_ боломжтой. Хэрвээ нэг warp доторх хоёр thread өөр өөр үйлдэл хийхээр заавар авбал, та үнэндээ _хоёр_ үйлдлийг хоёуланг нь хийж, шаардлагагүй цөмүүдийг масклаж, зөвхөн хэрэгтэй цөмүүдийг ажиллуулдаг.

{% include figure.liquid path="assets/gpu/warp-divergence.png" class="img-fluid" caption="<b>Зураг:</b> Thread-үүдийн багц дотор warp divergence-ийн жишээ (<a href='https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf'>эх сурвалж</a>). Цагаан зайнууд нь зарим физик CUDA цөмүүд зогсолт хийж байгааг харуулна." %}

Энэ нь thread түвшинд уян хатан програмчлал хийх боломжийг олгодог, гэхдээ хэрвээ warps хэтэрхий олон удаа салвал гүйцэтгэл чимээгүйгээр мууддаг. Threads нь бас ямар санах ойд хандаж болох талаар илүү уян хатан байдаг; VPU нь зөвхөн дараалсан санах ойн блокууд дээр ажиллаж чаддаг бол, CUDA cores нь shared registers доторх тус тусын float-д хандаж, thread бүрийн төлөвийг хадгалж чаддаг.

**CUDA цөмийн төлөвлөлт илүү уян хатан байдаг:** SM-ууд олон утаслагдсан CPU шиг ажилладаг, өөрөөр хэлбэл тэд олон програмуудыг (**warps**) зэрэг (нэг SM дээр 64 хүртэл) "төлөвлөж" чадна, гэхдээ _Warp Scheduler_ нь зөвхөн нэг програмыг нэг цагийн мөчлөгт гүйцэтгэдэг.<d-footnote>Тухайн SM дээр төлөвлөгдсөн warps-ыг "оршин суугч" гэж нэрлэдэг.</d-footnote> Warp Scheduler нь идэвхтэй warps хооронд автоматаар шилжиж, санах ойн ачаалал зэрэг I/O үйлдлүүдийг нууж өгдөг. TPU-уудтай харьцуулахад ихэвчлэн нэг утаслагдсан байдаг.

### Санах ой

Тооцооллын нэгжүүдээс гадна, GPU-д санах ойн шатлал байдаг. Хамгийн том нь HBM (гол GPU санах ой), дараа нь жижиг санах ойн кэшүүд (L2, L1/SMEM, TMEM, бүртгэлийн санах ой) байна.

* **Регистрүүд:** Тус бүрийн дэд хэсэг өөрийн гэсэн регистрийн файлтай бөгөөд H100/B200 дээр 16,384 ширхэг 32-бит үгтэй байдаг (`4 * 16384 * 4 = 256kiB` нэг SM бүрт) бөгөөд эдгээрийг CUDA цөмүүд ашиглаж болно.
  * Тус бүрийн CUDA цөм нэг удаад 256 регистр л ашиглаж чадна. Тиймээс бид нэг SM дээр 64 "оршин байгаа warp"-ыг төлөвлөж болох ч, хэрвээ нэг thread бүр 256 регистр ашиглавал нэг удаад зөвхөн 8 (`256 * 1024 / (4 * 32 * 256)`) багтааж чадна.

* **SMEM (L1 Cache):** тус бүр SM өөрийн 256кБ хэмжээтэй чипэн дээрх кэштэй бөгөөд үүнийг SMEM гэж нэрлэдэг. Энэ нь программистын удирдлагаар "shared memory" буюу хуваалцсан санах ой болж эсвэл техник хангамжаар чипэн дээрх кэш болж ашиглагдаж болно. SMEM нь идэвхжүүлэлт болон TC matmuls-д орох оролтуудыг хадгалахад ашиглагддаг.

* **L2 Cache:** бүх SM-ууд хуваалцдаг<d-footnote>Техникийн хувьд, L2 cache хоёр хэсэгт хуваагддаг, тиймээс H100 дээр хагас SM-ууд тус бүр 25MB-д хандаж чадна. Хоёр хэсгийг холбосон холбоос байдаг ч энэ нь бага хурдтай.</d-footnote> харьцангуй том ~50MB L2 cache бөгөөд энэ нь үндсэн санах ойд хандахыг багасгахад хэрэглэгддэг.
  * Энэ нь хэмжээний хувьд TPU-ийн VMEM-тэй төстэй боловч **илүү** удаан бөгөөд программист удирдаж чадахгүй. Энэ нь "хол зайд байгаа нууцлаг үйлдэл" шиг байдал үүсгэдэг. Программист L2 cache-ийг сайн ашиглахын тулд санах ойн хандалтын хэв маягийг өөрчлөх хэрэгтэй болдог.<d-footnote>L2 cache бүх SM-ууд дээр хуваалцагддаг тул программист SM-уудыг зохицуулалттай ажиллуулах шаардлагатай болдог. Үнэндээ эдгээр нь бие даасан нэгжүүд боловч.</d-footnote>
  * NVIDIA компани өөрийн чипүүдийн L2 cache-ийн дамжуулах хурдыг нийтэлдэггүй, гэхдээ [судалгаагаар](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth) ойролцоогоор 5.5TB/с гэж хэмжигдсэн. Энэ нь HBM-ийн дамжуулах хурднаас 1.6 дахин их боловч энэ нь хоёр чиглэлд (full-duplex) ажилладаг тул бодит хоёр талын дамжуулах хурд нь 3 дахин их байдаг. Харьцуулбал, TPU-ийн VMEM нь 2 дахин том *мөн* илүү их дамжуулах хурдтай (ойролцоогоор 40TB/с).

* **HBM:** GPU-ийн үндсэн санах ой, энэ нь model-ийн weights, gradients, activations гэх мэт зүйлсийг хадгалдаг.
  * HBM-ийн хэмжээ Volta дээр 32GB байсан бол Blackwell (B200) дээр 192GB болж ихэссэн.
  * HBM-ээс CUDA Tensor Core руу дамжих хурдыг HBM bandwidth эсвэл memory bandwidth гэж нэрлэдэг, энэ нь H100 дээр ойролцоогоор 3.35TB/s, B200 дээр 9TB/s байдаг.

### GPU-ийн үзүүлэлтийн хураангуй

Энд сүүлийн үеийн GPU-уудын техникийн үзүүлэлтийн товч танилцуулга байна. Нэг GPU-ийн хувилбаруудын SM-ийн тоо, цагийн хурд (clock speed), FLOPs нь бага зэрэг ялгаатай байдаг. Энд санах ойн багтаамжийн тоонууд байна:

|  GPU  | Үе шат (Generation) |   Цагийн хурд (Clock Speed)   | SMs/чип | SMEM багтаамж/SM | L2 багтаамж/чип | HBM багтаамж/чип |
| :---: | :----------------: | :--------------------------: | :-----: | :---------------: | :--------------: | :----------------: |
| V100  |   Volta            | 1.25GHz/1.38HGz              |    80   |       96kB        |       6MB        |       32GB         |
| A100  |   Ampere           | 1.10GHz/1.41GHz              |   108   |      192kB        |      40MB        |       80GB         |
| H100  |   Hopper           | 1.59GHz/1.98GHz              |   132   |      256kB        |      50MB        |       80GB         |
| H200  |   Hopper           | 1.59GHz/1.98GHz              |   132   |      256kB        |      50MB        |      141GB         |
| B200  | Blackwell          |        ?                     |   148   |      256kB        |     126MB        |      192GB         |

Бүх үеийнхэнд SM бүрт 256кB register санах ой байдаг. Blackwell нь SM бүрт 256кB TMEM нэмсэн. Энд чип бүрийн FLOPs болон дамжуулах зурвасын тоонууд байна:

|  GPU  | Үе шат | HBM BW/чип | FLOPs/сек/чип (bf16/fp16) | FLOPs/сек/чип (fp8/int8) | FLOPs/сек/чип (fp4) |
| :---: | :-----: | :--------: | :-----------------------: | :----------------------: | :-----------------: |
| V100  |  Volta  |   9.0e11   |            —              |            —             |         —           |
| A100  | Ampere  |   2.0e12   |          3.1e14           |         6.2e14           |         —           |
| H100  | Hopper  |   3.4e12   |          9.9e14           |         2.0e15           |         —           |
| H200  | Hopper  |   4.8e12   |          9.9e14           |         2.0e15           |         —           |
| B200  | Blackwell | 8.0e12    |          2.3e15           |         4.5e15           |       9.0e15        |

Бид B100-г оруулахгүй, учир нь энэ нь их хэмжээгээр үйлдвэрлэгдээгүй.<d-footnote>NVIDIA B100 үеийг бүтээсэн боловч тэдгээрийг маш богино хугацаанд зарж, үйлдвэрлэсэн. Үүний шалтгаан нь зарим эх сурвалжаар загварын алдаа байсан бөгөөд энэ нь тэднийг зарласан үзүүлэлтэд ойртож ажиллах боломжгүй болгосон. Хэт халалт болон цахилгаан зарцуулалтын асуудлаас болж хамгийн их FLOPs-д хүрч чадаагүй, хурд нь багассан.</d-footnote> Зарим үзүүлэлт нь GPU-ийн яг ямар хувилбараас хамаарч бага зэрэг өөр байж болно, учир нь NVIDIA GPU-ууд TPU шиг стандарт биш.

Энд GPU ба TPU бүрэлдэхүүн хэсгүүдийг харьцуулсан хэрэгтэй cheat sheet байна:

|              GPU              |     TPU     |               Энэ юу вэ?               |
| :---------------------------: | :---------: | :------------------------------------: |
| Streaming Multiprocessor (SM) | Tensor Core | Бусад нэгжүүдийг агуулсан үндсэн "эс"  |
|        Warp Scheduler         |     VPU     |      SIMD вектор тооцооллын нэгж      |
|           CUDA Core           |   VPU ALU   |               SIMD ALU                 |
|        SMEM (L1 Cache)        |    VMEM     |       Чипэн доторх хурдан кэш санах ой  |
|          Tensor Core          |     MXU     |      Матриц үржүүлэх нэгж              |
|        HBM (aka GMEM)         |     HBM     |  Өндөр хурдтай, их багтаамжтай санах ой |

### GPU-ууд ба TPU-ууд чипийн түвшинд

GPU-ууд анх видео тоглоомын дүрсийг зурж эхэлсэн, гэхдээ 2010-аад оноос гүнзгий сургалт (deep learning) их хөгжсөнөөс хойш тэд илүү их матриц үржүүлэх зориулалттай машин шиг ажиллаж эхэлсэн – өөрөөр хэлбэл, TPU шиг болсон.<d-footnote>Гүнзгий сургалтын өсөлтөөс өмнө GPU ("Graphics Processing Units") нь зөвхөн дүрс зурдаг байсан – ихэвчлэн видео тоглоомд хэрэглэдэг. Видео тоглоомд объектуудыг сая сая жижиг гурвалжнаар дүрсэлдэг, тоглоом эдгээр гурвалжнуудыг 2D зураг болгон хувиргаж (эсвэл "растеризаци" хийж) дэлгэц дээр секундэд 30-60 удаа харуулдаг (энэ давтамжийг фреймрейт гэдэг). Растеризаци гэдэг нь эдгээр гурвалжнуудыг камерын координатын системд шилжүүлж, аль гурвалжин аль пикселтэй давхцаж байгааг тооцоолохыг хэлнэ, энэ нь секундэд тэрбум тэрбумаар тооцоологддог. Та төсөөлж байгаачлан, энэ нь маш их зардалтай, бас зөвхөн эхлэл нь. Дараа нь та тус бүр пикселийг өнгөөр будах хэрэгтэй, магадгүй хэд хэдэн хагас тунгалаг гурвалжингуудын өнгийг нийлүүлж будах шаардлагатай. GPU-ууд ийм үйлдлүүдийг маш хурдан хийхээр бүтээгдсэн, бас олон төрлийн ажил ("shader" гэж нэрлэдэг) зэрэг гүйцэтгэх хэрэгтэй болдог, нэг ч үйлдэл давамгайлдаггүй. Үүний үр дүнд, хэрэглэгчдэд зориулсан график GPU-ууд матриц үржүүлэлт хийж чаддаг ч энэ нь тэдний үндсэн үүрэг биш юм.</d-footnote> Энэ түүх нь орчин үеийн GPU-ууд яагаад ийм харагддагийг тодорхой хэмжээгээр тайлбарлаж өгдөг. Тэд зөвхөн LLM эсвэл ML загварт зориулагдаагүй, харин ерөнхий зориулалтын хурдасгуур болгон бүтээгдсэн, мөн техник хангамж нь "ерөнхий" байхыг зорьдог нь заримдаа ашигтай, заримдаа асуудалтай болдог. GPU ихэвчлэн шинэ даалгаварт "шууд ажилладаг" бөгөөд TPU-тай харьцуулахад сайн компайлераас бага хамаардаг. Гэхдээ энэ нь тэднийг ойлгоход илүү хэцүү, эсвэл хамгийн сайн гүйцэтгэл гаргахад төвөгтэй болгодог, учир нь олон компайлерийн боломжууд саатал үүсгэж болдог.

**GPU-ууд илүү модульчлагдсан байдаг.** TPU-д 1-2 том Tensor Core байдаг бол GPU-д хэдэн зуун жижиг SM байдаг. Мөн адил, тус бүр Tensor Core нь 4 том VPU-тэй бөгөөд тус бүр 1024 ALU-тай, харин GPU-ууд (жишээ нь H100) нь 132 * 4 = 528 жижиг, бие даасан SIMD нэгжтэй. Энэ санааг тодруулахын тулд GPU ба TPU-г 1:1 харьцуулсан хүснэгт энд байна:

|              GPU              |           TPU            | H100 # | TPU v5p # |
| :---------------------------: | :----------------------: | :----: | :-------: |
| SM (streaming multiprocessor) |       Tensor Core        |  132   |     2     |
|        Warp Scheduler         |           VPU            |  528   |     8     |
|        SMEM (L1 cache)        |           VMEM           |  32MB  |   128MB   |
|           Registers           | Vector Registers (VRegs) |  32MB  |   256kB   |
|          Tensor Core          |           MXU            |  528   |     8     |

Энэ модуляр байдлын ялгаа нь нэг талаас TPU-г хийхэд илүү хямд, ойлгоход хялбар болгодог. Гэхдээ энэ нь compiler-д зөв зүйл хийхэд илүү их ачаалал өгдөг. TPU нь нэг л удирдлагын thread-тэй бөгөөд зөвхөн vector-чилсэн VPU-wide заавар дэмждэг тул compiler нь бүх санах ойн ачаалал болон MXU/VPU ажлыг гараар pipeline хийх хэрэгтэй болдог, ингэснээр саатал үүсэхээс сэргийлнэ. GPU программист олон өөр kernel-уудыг ажиллуулж чадна, эдгээр нь бүгд тусдаа SM дээр бие даан ажиллана. Нөгөө талаас, эдгээр kernel-ууд L2 cache-ийг хэт их ашиглах эсвэл санах ойн ачааллыг нэгтгэж чадахгүйгээс болж гүйцэтгэл муу байж магадгүй; hardware нь runtime-ийн ихэнх хэсгийг удирддаг учраас, дотоодод яг юу болж байгааг ойлгоход хэцүү болдог. Үүний үр дүнд, TPU нь ихэнхдээ дээд roofline гүйцэтгэлд бага ажил хийж хүрч чаддаг.

**Түүхэн талаасаа, нэг бүрийн GPU нь ижил төрлийн TPU-гаас илүү хүчтэй (мөн илүү үнэтэй) байдаг:** Нэг H200 нь TPU v5p-ээс бараг 2 дахин их FLOPs/s, мөн 1.5 дахин их HBM-тэй. Үүний зэрэгцээ, Google Cloud дээрх үнэ нь TPU v5p-д ойролцоогоор \\$10/hour for an H200 compared to \\$4/цаг байдаг. TPU-ууд нь ихэвчлэн олон чипийг сүлжээгээр холбох дээр илүү их найдаж ажилладаг бол GPU-ууд тийм биш.

**TPU-д илүү хурдан кэш санах ой их байдаг.** TPU-д VMEM илүү их байдаг, харин GPU-д SMEM (+TMEM) байдаг. Энэ санах ойг жингүүд болон идэвхжүүлэлтийг хадгалахад ашиглаж болно. Ингэснээр эдгээрийг маш хурдан ачааллаж, ашиглах боломжтой болдог. Хэрвээ та VMEM-д моделийн жингүүдийг тогтмол хадгалах эсвэл урьдчилан ачаалж чадвал, энэ нь LLM inference-ийг илүү хурдан болгодог.

### Шалгалт 1: GPU техник хангамж

Энд дээрх агуулгыг шалгах зарим дасгал байна. Хариултуудыг өгсөн байгаа, гэхдээ асуултуудыг эхлээд өөрөө хариулахыг хичээвэл сайн. Үзэг цаас бэлэн байлгаарай.

**Асуулт 1 [CUDA цөмүүд]:** H100-д хэдэн fp32 CUDA цөм (ALU) байдаг вэ? B200-д хэд вэ? Энэ нь TPU v5p-ийн бие даасан ALU-уудын тоотой хэрхэн харьцдаг вэ?

{% details Хариуг харахын тулд энд дарна уу. %}

**Хариулт:** H100 нь 132 SM-тэй бөгөөд тус бүр нь 4 дэд хэсэгтэй, дэд хэсэг бүрт 32 fp32 CUDA цөмтэй, тэгэхээр бидэнд `132 * 4 * 32 = 16896` CUDA цөм байна. B200 нь `148` SM-тэй, тэгэхээр нийт `18944`. TPU v5p нь 2 TensorCore-той (ихэвчлэн Megacore-оор холбогдсон), тус бүр нь VPU-тай бөгөөд (8, 128) сувгуудтай, сувгийн бүрт 4 тусдаа ALU-тай, тэгэхээр `2 * 4 * 8 * 128 = 8192` ALU байна. Энэ нь H100-ийн вектор сувгийн тооны ойролцоогоор тал хувьтай, ойролцоо давтамжтай ажилладаг.

{% enddetails %}

**Асуулт 2 [Вектор FLOPs тооцоолол]**: Нэг H100 нь 132 SM-тэй бөгөөд 1.59GHz цагийн хурдтай ажилладаг (1.98GHz хүртэл boost хийж болно). Нэг ALU бүр цикл тутамд нэг вектор үйлдэл хийж чадна гэж үзье. Нэг секундэд хэдэн вектор fp32 FLOPs хийж чадах вэ? Boost-тай үед хэд вэ? Энэ нь матриц үржүүлэх (matmul) FLOPs-тай харьцуулахад ямар вэ?

{% details Хариуг харахын тулд энд дарна уу. %}

**Хариулт:** `132 * 4 * 32 * 1.59e9 = 26.9TFLOPs/s`. Boost ашиглавал энэ нь 33.5 TFLOPs/s байна. Энэ нь [spec sheet](https://www.nvidia.com/en-us/data-center/h100/)-д бичсэнээс хоёр дахин бага байна, учир нь техникийн хувьд бид нэг циклд FMA (fused-multiply-add) хийж чадна, энэ нь хоёр FLOPs гэж тооцогдоно, гэхдээ ихэнх тохиолдолд энэ нь хэрэгтэй биш. Бид 990 bfloat16 matmul TFLOPs/s хийж чадна, тэгэхээр FMA-г тооцохгүй бол Tensor Cores нь ойролцоогоор 30 дахин их FLOPs/s хийдэг.

{% enddetails %}

**Асуулт 3 [GPU matmul intensity]:** H100 дээр fp16 matmul-ийн дээд хүч чадал (intensity) хэд вэ? B200 дээр хэд вэ? fp8 дээр хэд вэ? *Intensity гэдэг нь matmul FLOPs/секунд-ийг санах ойн дамжуулалтын өргөн зурвасын харьцаагаар хэмждэг.*

{% details Хариуг харахын тулд энд дарна уу. %}

**Хариулт:** H100-д бидэнд оргил 990e12 fp16 FLOPs болон 3.35e12 байт / секундын дамжуулалтын өргөн байна. Тиймээс чухал интенсив нь `990e12 / 3.35e12 = 295`, TPU-гийн 240-тай ойролцоо байна. B200-д энэ нь `2250e12 / 8e12 = 281`, мөн адилхан байна. Энэ нь TPU-тай төстэйгээр, матмул дээр тооцооллын хязгаарт хүрэхийн тулд ойролцоогоор 280 багц хэмжээ хэрэгтэй гэсэн үг юм.

H100 болон B200-д хоёуланд нь яг 2x fp8 FLOPs байгаа, тиймээс хамгийн их intensity нь мөн хоёр дахин нэмэгдэж 590 болон 562 болж байна. Гэхдээ нэг талаасаа энэ нь тогтвортой хэвээр байна гэж хэлж болно, учир нь бидний weights ихэнхдээ fp8 хэлбэрээр ачаалагдах магадлалтай.

{% enddetails %}

**Асуулт 4 [Matmul runtime]:** 3-р асуултын хариуг ашиглан, нэг B200 дээр `fp16[64, 4096] * fp16[4096, 8192]` матмул хийхэд хэр удаан хугацаа зарцуулах вэ? Харин `fp16[512, 4096] * fp16[4096, 8192]` дээр бол яах вэ?

{% details Хариуг харахын тулд энд дарна уу. %}

Дээрхээс харахад, бид 281 токенээс бага batch size-тай үед харилцааны хязгаарлалттай байна. Тиймээс эхний тохиолдолд зөвхөн bandwidth-ийн хязгаарлалттай байна. Бид $2BD + 2DF + 2BF$ байт (`2*64*4096 + 2*4096*8192 + 2*64*8192=69e6`)-ийг `8e12` байт/секунд bandwidth-тайгаар уншиж эсвэл бичдэг, тэгэхээр энэ нь ойролцоогоор `69e6 / 8e12 = 8.6us` хугацаа шаардана. Бодит амьдрал дээр бид нийт bandwidth-ийн зөвхөн нэг хэсгийг л авдаг тул энэ нь 10-12 микросекунд (us) орчим үргэлжилж магадгүй. Batch size-ийг нэмэхэд бид бүрэн compute-bound буюу тооцооллын хязгаарлалттай болдог, тиймээс бид `T=2*512*4096*8192/2.3e15=15us` гэж хүлээнэ. Мөн бид нийт FLOPs-ийн зөвхөн нэг хэсгийг л ашиглах тул энэ нь 20 микросекунд (us) орчим байж магадгүй.

{% enddetails %}

**Асуулт 5 [L1 cache багтаамж]:** H100-д нийт L1/SMEM багтаамж хэд вэ? Register memory хэд вэ? Энэ нь TPU VMEM багтаамжтай хэрхэн харьцдаг вэ?

{% details Хариуг харахын тулд энд дарна уу. %}

**Хариулт:** Бидэнд нэг SM бүрт 256kB SMEM болон 256kB register санах ой бий, тэгэхээр тус бүр ойролцоогоор 33MB (`132 * 256kB`) байна. Хоёуланг нь нийлүүлбэл нийтдээ ойролцоогоор 66MB болно. Энэ нь орчин үеийн TPU-гийн VMEM-ийн 120MB-ийн тал хувьтай тэнцэнэ, гэхдээ TPU-д нийтдээ зөвхөн 256kB register санах ой байдаг! TPU VMEM-ийн хүлээлгийн хугацаа (latency) нь SMEM-ийнхээс бага байдаг, энэ нь TPU дээр register санах ой тийм ч чухал биш байдаг нэг шалтгаан юм (VMEM рүү өгөгдөл шилжүүлэх нь хямдхан).

{% enddetails %}

**Асуулт 6 [B200 цагийн давтамжийг тооцоолох]:** NVIDIA [энд](https://resources.nvidia.com/en-us-blackwell-architecture) мэдээлснээр B200 нь вектор fp32 тооцоололд 80TFLOPs/секунд гүйцэтгэж чадна. Хэрвээ тус бүр CUDA цөм нь FMA (fused multiply add) үйлдэлд 1 циклд 2 FLOPs хийж чаддаг бол, дээд талын цагийн давтамжийг тооцоолоорой.

{% details Хариуг харахын тулд энд дарна уу. %}

**Хариулт:** Бидэнд 148 * 4 * 32 = 18944 CUDA цөм байгаа гэдгийг мэднэ, тиймээс бид `18944 * 2 = 37888 FLOPs / cycle` хийж чадна. Тиймээс `80e12 / 37888 = 2.1GHz`, энэ нь өндөр боловч боломжийн дээд давтамж юм. B200-ууд ихэвчлэн шингэн хөргөлттэй байдаг тул өндөр давтамж илүү боломжийн юм.

{% enddetails %}

**Асуулт 7 [H100 дээр нэмэх ажиллагааны хугацааг тооцоолох]:** Дээрх зураглалуудыг ашиглан, хоёр `fp32[N]` векторыг нэг H100 дээр нэмэхэд хэр хугацаа шаардагдахыг тооцоолно уу. $T_\text{math}$ болон $T_\text{comms}$-г хоёуланг нь тооцоолно уу. Энэ үйлдлийн arithmetic intensity хэд вэ? Хэрвээ боломжтой бол, энэ үйлдлийг PyTorch эсвэл JAX дээр ажиллуулж `N = 1024` болон `N=1024 * 1024 * 1024`-г шалгаарай. Үр дүн нь хэрхэн ялгаатай байна вэ?

{% details Хариуг харахын тулд энд дарна уу. %}

**Хариулт:** Эхлээд, хоёр `fp32[N]` вектор нэмэхэд N FLOPs хийгдэнэ ба `4 * N * 2` байт уншигдаж, 4 * N байт буцаан бичигдэнэ, нийтдээ `3 * 4 * N = 12N` болно. Тэдний харьцааг тооцвол `total FLOPs / total bytes = N / 12N = 1 / 12` гарна, энэ нь нэлээд муу үр дүнтэй байна.

Бид дээр тооцоолсончлон, бид ойролцоогоор 33.5 TFLOPs/секунд хурдтай ажиллах боломжтой, FMA-г тооцохгүйгээр. Энэ нь зөвхөн бүх CUDA цөмүүдийг ашигласан тохиолдолд л боломжтой. `N = 1024` хувьд бид хамгийн ихдээ 1024 CUDA цөм буюу 8 SM-г ашиглаж чадна, энэ нь илүү удаан (ойролцоогоор 16 дахин удаан, хэрвээ бид тооцоололд хязгаарлагдсан бол) болно. Мөн бид 3.35e12 байт/секунд санах ойн зурвасын өргөнтэй. Тиймээс бидний хамгийн их техник хангамжийн эрчим нь `33.5e12 / 3.35e12 = 10` байна.<d-footnote>Энэ эрчим нь сүүлийн үеийн GPU-уудын хувьд тогтвортой байдаг нь анхаарал татахуйц. H100-ийн хувьд энэ нь 33.5 / 3.5, B200-ийн хувьд 80 / 8 байна. Яагаад ийм байгаа нь тодорхой биш ч сонирхолтой ажиглалт юм.</d-footnote> Тиймээс бид маш их холбооны хязгаарлалтад орно. Тиймээс бидний ажиллах хугацаа нь зүгээр л

$$T = \max(T_\text{comms}, T_\text{math}) = \frac{12 \cdot N}{\text{3.35e12}} = \frac{N}{\text{2.8e11}}$$

For `N = 65,536`, this is about 0.23us. In practice we see a runtime of about 1.5us in JAX, which is fine because we expect to be super latency bound here. For `N = 1024 * 1024 * 1024`, we have a roofline of about 3.84ms, and we see 4.1ms, which is good!

{% enddetails %}

## Networking

Networking is one of the areas where GPUs and TPUs differ the most. As we’ve seen, TPUs are connected in 2D or 3D tori, where each TPU is only connected to its neighbors. This means sending a message between two TPUs must pass through every intervening TPU, and forces us to use only uniform communication patterns over the mesh. While inconvenient in some respects, this also means the number of links per TPU is constant and we can scale to arbitrarily large TPU "pods" without loss of bandwidth.

GPUs on the other hand use a more traditional hierarchical tree-based switching network. Sets of 8 GPUs called **nodes** (up to 72 for GB200<d-footnote>The term node is overloaded and can mean two things: the NVLink domain, aka the set of GPUs fully connected over NVLink interconnects, or the set of GPUs connected to a single CPU host. Before B200, these were usually the same, but in GB200 NVL72, we have an NVLink domain with 72 GPUs but still only 8 GPUs connected to each host. We use the term node here to refer to the NVLink domain, but this is controversial.</d-footnote>) are connected within 1 hop of each other using high-bandwidth interconnects called NVLinks, and these nodes are connected into larger units (called **SUs** or Scalable Units) with a lower bandwidth InfiniBand (IB) or Ethernet network using NICs attached to each GPU. These in turn can be connected into arbitrarily large units with higher level switches.

{% include figure.liquid path="assets/gpu/superpod-diagram.png" class="img-fluid" caption="<b>Figure:</b> a diagram showing a typical H100 network. A set of 8 GPUs is connected into a node or NVLink domain with NVSwitches (also called NVLink switches), and these nodes are connected to each other with a switched InfiniBand fabric. H100s have about 450GB/s of egress bandwidth each in the NVLink domain, and each node has 400GB/s of egress bandwidth into the IB network." %}

### At the node level

A GPU node is a small unit, typically of 8 GPUs (up to 72 for GB200), connected with all-to-all, full-bandwidth, low latency NVLink interconnects.<d-footnote>NVLink has been described to me as something like a souped-up PCIe connection, with low latency and protocol overhead but not designed for scalability/fault tolerance, while InfiniBand is more like Ethernet, designed for larger lossy networks.</d-footnote> Each node contains several high-bandwidth NVSwitches which switch packets between all the local GPUs. The actual node-level topology has changed quite a bit over time, including the number of switches per node, but for H100, we have 4 NVSwitches per node with GPUs connected to them in a `5 + 4 + 4 + 5` link pattern, as shown:

{% include figure.liquid path="assets/gpu/nvlink-nodes.png" class="img-fluid" caption="<b>Figure:</b> node aka NVLink domain diagrams from Pascall (P100) onward. Since Volta (V100), we have had all-to-all connectivity within a node using a set of switches. The H100 node has 4 NVSwitches connected to all 8 GPUs with 25GB/s links." %}

For the Hopper generation (NVLink 4.0), each NVLink link has 25GB/s of full-duplex<d-footnote>Full-duplex here means 25GB/s each way, with both directions independent of each other. You can send a total of 50GB/s over the link, but at most 25GB/s in each direction.</d-footnote> bandwidth (50GB/s for B200), giving us `18 * 25=450GB/s` of full-duplex bandwidth from each GPU into the network. The massive NVSwitches have up to 64 NVLink ports, meaning an 8xH100 node with 4 switches can handle up to `64 * 25e9 * 4=6.4TB/s` of bandwidth. Here’s an overview of how these numbers have changed with GPU generation:

| NVLink Gen | NVSwitch Gen | GPU Generation | NVLink Bandwidth (GB/s, full-duplex) | NVLink Ports / GPU | Node GPU to GPU bandwidth (GB/s full-duplex) | Node size (NVLink domain) | NVSwitches per node |
| :--------: | :----------: | :------------: | :----------------------------------: | :----------------: | :------------------------------------------: | :-----------------------: | :-----------------: |
|  **3.0**   |   **2.0**    |     Ampere     |                  25                  |         12         |                     300                      |             8             |          6          |
|  **4.0**   |   **3.0**    |     Hopper     |                  25                  |         18         |                     450                      |             8             |          4          |
|  **5.0**   |   **4.0**    |   Blackwell    |                  50                  |         18         |                     900                      |           8/72            |        2/18         |

Blackwell (B200) has nodes of 8 GPUs. GB200NVL72 support larger NVLink domains of 72 GPUs. We show details for both the 8 and 72 GPUs systems.

### Quiz 2: GPU nodes

Here are some more Q/A problems on networking. I find these particularly useful to do out, since they make you work through the actual communication patterns.

**Question 1 [Total bandwidth for H100 node]:** How much total bandwidth do we have per node in an 8xH100 node with 4 switches? *Hint:* consider both the NVLink and NVSwitch bandwidth.

{% details Click here for the answer. %}

**Answer:** We have Gen4 4xNVSwitches, each with `64 * 25e9=1.6TB/s` of unidirectional bandwidth. That would give us `4 * 1.6e12=6.4e12` bandwidth at the switch level. However, note that each GPU can only handle 450GB/s of unidirectional bandwidth, so that means we have at most `450e9 * 8 = 3.6TB/s` bandwidth. Since this is smaller, the peak bandwidth is 3.6TB/s.

{% enddetails %}

**Question 2 [Bisection bandwidth]**: Bisection bandwidth is defined as the smallest bandwidth available between any even partition of a network. In other words, if split a network into two equal halves, how much bandwidth crosses between the two halves? Can you calculate the bisection bandwidth of an 8x H100 node? *Hint:* bisection bandwidth typically includes flow in both directions.

{% details Click here for the answer. %}

**Answer:** Any even partition will have 4 GPUs in each half, each of which can egress `4 * 450GB/s` to the other half. Taking flow in both directions, this gives us `8 * 450GB/s` of bytes cross the partition, or 3.6TB/s of bisection bandwidth. This is what NVIDIA reports e.g. [here](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf).

{% enddetails %}

**Question 3 [AllGather cost]**: Given an array of B bytes, how long would a (throughput-bound) AllGather take on an 8xH100 node? Do the math for bf16[D<sub>X</sub>, F] where `D=4096`, `F=65,536`. *It’s worth reading the TPU collectives [section](https://jax-ml.github.io/scaling-book/sharding/) before answering this. Think this through here but we’ll talk much more about collectives next.*

{% details Click here for the answer. %}

**Answer:** Each GPU can egress 450GB/s, and each GPU has $B / N$ bytes (where `N=8`, the node size). We can imagine each node sending its bytes to each of the other $N - 1$ nodes one after the other, leading to a total of (N - 1) turns each with $T_\text{comms} = (B / (N * W_\text{unidirectional}))$, or $T_\text{comms} = (N - 1) * B / (N * W_\text{unidirectional})$. This is approximately $B / (N * W_\text{uni})$ or $B / \text{3.6e12}$, the bisection bandwidth.

For the given array, we have `B=4096 * 65536 * 2=512MB`, so the total time is `536e6 * (8 - 1) / 3.6e12 = 1.04ms`. This could be latency-bound, so it may take longer than this in practice (in practice it takes about 1.5ms).

{% enddetails %}

## Beyond the node level

Beyond the node level, the topology of a GPU network is less standardized. NVIDIA publishes a [reference DGX SuperPod architecture](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html) that connects a larger set of GPUs than a single node using InfiniBand, but customers and datacenter providers are free to customize this to their needs.<d-footnote>For instance, Meta trained LLaMA-3 on a datacenter network that differs significantly from this description, using Ethernet, a 3 layer switched fabric, and an oversubscribed switch at the top level.</d-footnote>

Here is a diagram for a reference 1024 GPU H100 system, where each box in the bottom row is a single 8xH100 node with 8 GPUs, 8 400Gbps CX7 NICs (one per GPU), and 4 NVSwitches.

{% include figure.liquid path="assets/gpu/h100-superpod.png" class="img-fluid" caption="<b>Figure:</b> diagram of the reference 1024 H100 DGX SuperPod with 128 nodes (sometimes 127), each with 8 H100 GPUs, connected to an InfiniBand scale-out network. Sets of 32 nodes (256 GPUs) are called 'Scalable Units' or SUs. The leaf and spine IB switches provide enough bandwidth for full bisection bandwidth between nodes." %}

**Scalable Units:** Each set of 32 nodes is called a "Scalable Unit" (or SU), under a single set of 8 leaf InfiniBand switches. This SU has 256 GPUs with 4 NVSwitches per node and 8 Infiniband leaf switches. All the cabling shown is InfiniBand NDR (50GB/s full-duplex) with 64-port NDR IB switches (also 50GB/s per port). *Note that the IB switches have 2x the bandwidth of the NVSwitches (64 ports with 400 Gbps links).*

**SuperPod:** The overall SuperPod then connects 4 of these SUs with 16 top level "spine" IB switches, giving us 1024 GPUs with 512 node-level NVSwitches, 32 leaf IB switches, and 16 spine IB switches, for a total of 512 + 32 + 16 = 560 switches. Leaf switches are connected to nodes in sets of 32 nodes, so each set of 256 GPUs has 8 leaf switches. All leaf switches are connected to all spine switches.

**How much bandwidth do we have?** The overall topology of the InfiniBand network (called the "scale out network") is that of a **fat tree**, with the cables and switches guaranteeing full bisection bandwidth above the node level (here, 400GB/s). That means if we split the nodes in half, each node can egress 400GB/s to a node in the other partition at the same time. More to the point, this means we should have a roughly constant AllReduce bandwidth in the scale out network! While it may not be implemented this way, you can imagine doing a ring reduction over arbitrarily many nodes in the scale-out network, since you can construct a ring including every one.

| Level | GPUs | Switches per Unit | Switch Type | Bandwidth per Unit (TB/s, full-duplex) | GPU-to-GPU Bandwidth (GB/s, full-duplex) | Fat Tree Bandwidth (GB/s, full-duplex) |
| :---: | :------------: | :-------------------------: | :---------: | :------------------------------------------: | :--------------------------------------: | :---: |
| Node  |       8        |              4              |     NVL     |                     3.6                      |                   450                    | 450
| Leaf  |      256       |              8              |     IB      |                     12.8                     |                    50                    | 400 |
| Spine |      1024      |             16              |     IB      |                     51.2                     |                    50                    | 400 |

By comparison, a TPU v5p has about 90GB/s egress bandwidth per link, or 540GB/s egress along all axes of the 3D torus. This is not point-to-point so it can only be used for restricted, uniform communication patterns, but it still gives us a much higher TPU to TPU bandwidth that can scale to arbitrarily large topologies (at least up to 8960 TPUs).

The GPU switching fabric can in theory be extended to arbitrary sizes by adding additional switches or layers of indirection, at the cost of additional latency and costly network switches.

<p markdown=1 class="takeaway">**Takeaway**: Within an H100 node, we have a full fat tree bandwidth of 450GB/s from each GPU, while beyond the node, this drops to 400GB/s node-to-node. This will turn out to be critical for communication primitives.</p>

**GB200 NVL72s:** NVIDIA has recently begun producing new GB200 NVL72 GPU clusters that combine 72 GPUs in a single NVLink domain with full 900GB/s of GPU to GPU bandwidth. These domains can then be linked into larger SuperPods with proportionally higher (9x) IB fat tree bandwidth. Here is a diagram of that topology:

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>Figure:</b> a diagram showing a GB200 DGX SuperPod of 576 GPUs. Each rack at the bottom layer contains 72 GB200 GPUs." %}

Counting the egress bandwidth from a single node (the orange lines above), we have `4 * 18 * 400 / 8 = 3.6TB/s` of bandwidth to the leaf level, which is 9x more than an H100 (just as the node contains 9x more GPUs). That means the critical node egress bandwidth is much, _much_ higher and our cross-node collective bandwidth can actually be _lower_ than within the node.
See [Appendix A](#appendix-a-how-does-this-change-with-gb200) for more discussion.

|  Node Type  | GPUs per node | GPU egress bandwidth | Node egress bandwidth |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

<p markdown=1 class="takeaway">**Takeaway**: GB200 NVL72 SuperPods drastically increase the node size and egress bandwidth from a given node, which changes our rooflines significantly.</p>

### Quiz 3: Beyond the node level

**Question 1 [Fat tree topology]:** Using the DGX H100 diagram above, calculate the bisection bandwidth of the entire 1024 GPU pod at the node level. Show that the bandwidth of each link is chosen to ensure full bisection bandwidth. *Hint: make sure to calculate both the link bandwidth and switch bandwidth.*

{% details Click here for the answer. %}

**Answer:** Let’s do it component by component:

* First, each node has 8x400Gbps NDR IB cables connecting it to the leaf switches, giving each node `8 * 400 / 8 = 400 GB/s` of bandwidth to the leaf. We have 8 leaf switches with 3.2TB/s each (64 400 GBps links), but we can only use 32 of the 64 ports to ingress from the SU, so that’s `32 * 400 / 8 = 12.8TB/s` for 32 nodes, again exactly 400GB/s.
* Then at the spine level we have `8 * 16 * 2` 400Gbps NDR IB cables connecting each SU to the spine, giving each SU `8 * 16 * 2 * 400 / 8 = 12.8 TB/s` of bandwidth to the leaf. Again, this is 400GB/s per node. We have 16 spine switches, each with 3.2TB/s, giving us `16 * 3.2 = 51.2 TB/s`, which with 128 nodes is again 400GB/s.

Thus if we bisect our nodes in any way, we will have 400GB/s per GPU between them. Every component has exactly the requisite bandwidth to ensure the fat tree.

{% enddetails %}

**Question 2 [Scaling to a larger DGX pod]:** Say we wanted to train on 2048 GPUs instead of 1024. What would be the simplest/best way to modify the above DGX topology to handle this? What about 4096? *Hint: there’s no single correct answer, but try to keep costs down. Keep link capacity in mind. [This](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf) documentation may be helpful.*

{% details Click here for the answer. %}

**Answer:** One option would be to keep the SU structure intact (32 nodes under 8 switches) and just add more of them with more top-level switches. We’d need 2x more spine switches, so we’d have 8 SUs with 32 spine switches giving us enough bandwidth.

One issue with this is that we only have 64 ports per leaf switch, and we’re already using all of them in the above diagram. But instead it’s easy to do 1x 400 Gbps NDR cable per spine instead of 2x, which gives the same total bandwidth but saves us some ports.

For 4096 GPUs, we actually run out of ports, so we need to add another level of indirection, that is to say, another level in the hierarchy. NVIDIA calls these "core switches", and builds a 4096 GPU cluster with 128 spine switches and 64 core switches. You can do the math to show that this gives enough bandwidth.

{% enddetails %}

## How Do Collectives Work on GPUs?

GPUs can perform all the same collectives as TPUs: ReduceScatters, AllGathers, AllReduces, and AllToAlls. Unlike TPUs, the way these work changes depending on whether they’re performed at the node level (over NVLink) or above (over InfiniBand). These collectives are implemented by NVIDIA in the [NVSHMEM](https://developer.nvidia.com/nvshmem) and [NCCL](https://developer.nvidia.com/nccl) (pronounced "nickel") libraries. NCCL is open-sourced [here](https://github.com/NVIDIA/nccl). While NCCL uses a variety of implementations depending on latency requirements/topology ([details](https://github.com/NVIDIA/nccl/issues/1415#issuecomment-2310650081)), from here on, we’ll discuss a theoretically optimal model over a switched tree fabric.

### Intra-node collectives

**AllGather or ReduceScatter:** For an AllGather or ReduceScatter at the node level, you can perform them around a ring just like a TPU, using the full GPU-to-GPU bandwidth at each hop. Order the GPUs arbitrarily and send a portion of the array around the ring using the full GPU-to-GPU bandwidth.<d-footnote>You can also think of each GPU sending its chunk of size $\text{bytes} / N$ to each of the other $N - 1$ GPUs, for a total of $(N - 1) * N * bytes / N$ bytes communicated, which gives us</d-footnote> The cost of each hop is $T_\text{hop} = \text{bytes} / (N * \text{GPU egress bandwidth})$, so the overall cost is

$$T_\text{AG or RS comms} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot \text{GPU egress bandwidth}} \rightarrow \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

You’ll note this is exactly the same as on a TPU. For an AllReduce, you can combine an RS + AG as usual for twice the cost.

{% include figure.liquid path="assets/gpu/all-gather.gif" class="img-fluid" caption="<b>Figure:</b> bandwidth-optimal 1D ring AllGather algorithm. For B bytes, this sends V / X bytes over the top-level switches X - 1 times." %}

If you’re concerned about latency (e.g. if your array is very small), you can do a tree reduction, where you AllReduce within pairs of 2, then 4, then 8 for a total of $\log(N)$ hops instead of $N - 1$, although the total cost is still the same.

<p markdown=1 class="takeaway">**Takeaway:** the cost to AllGather or ReduceScatter an array of B bytes within a single node is about $T_\text{comms} = B * (8 - 1) / (8 * W_\text{GPU egress}) \approxeq B / W_\text{GPU egress}$. This is theoretically around $B  / \text{450e9}$ on an H100 and $B / \text{900e9}$ on a B200. An AllReduce has 2x this cost unless in-network reductions are enabled.</p>

<b markdown=1 style="color: #57cf57;">Pop Quiz 1 [AllGather time]:</b> Using an 8xH100 node with 450 GB/s full-duplex bandwidth, how long does AllGather(bf16[B<sub>X</sub>, F]) take? Let $B=1024$, $F=16,384$.

{% details Click here for the answer. %}

**Answer:** We have a total of $2 \cdot B \cdot F$ bytes, with 450e9 unidirectional bandwidth. This would take roughly $T_\text{comms} = (2 \cdot B \cdot F) / \text{450e9}$, or more precisely $(2 \cdot B \cdot F \cdot (8 - 1)) / (8 \cdot \text{450e9})$. Using the provided values, this gives us roughly $(2 \cdot 1024 \cdot 16384) / \text{450e9} = \text{75us}$, or more precisely, $\text{65us}$.

{% enddetails %}

**AllToAlls:** GPUs within a node have all-to-all connectivity, which makes AllToAlls, well, quite easy. Each GPU just sends directly to the destination node. Within a node, for B bytes, each GPU has $B / N$ bytes and sends $(B / N^2)$ bytes to $N - 1$ target nodes for a total of

$$T_\text{AllToAll comms} = \frac{B \cdot (N - 1)}{W \cdot N^2} \approx \frac{B}{W \cdot N}$$

Compare this to a TPU, where the cost is $B / (4W)$. Thus, within a single node, we get a 2X theoretical speedup in runtime ($B / 4W$ vs. $B / 8W$).

For Mixture of Expert (MoE) models, we frequently want to do a *sparse or ragged AllToAll,* where we guarantee at most $k$ of $N$ shards on the output dimension are non-zero, that is to say $T_\text{AllToAll} \rightarrow K[B, N]$ where at most $k$ of $N$ entries on each axis are non-zero. The cost of this is reduced by $k/N$, for a total of about $\min(k/N, 1) \cdot B / (W \cdot N)$. For an MoE, we often pick the non-zero values independently at random, so there's some chance of having fewer than $k$ non-zero, giving us approximately
$(N-1)/N \cdot \min(k/N, 1) \cdot B / (W \cdot N)$.<d-footnote>The true cost is actually $$(1 - \left(\frac{Z - 1}{Z}\right)^K) \cdot \frac{Z - 1}{Z}$$ the expected number of distinct outcomes in $K$ dice rolls, but it is very close to the approximation given. See the Appendix for more details.</d-footnote>

<b markdown=1 style="color: #c55404ff;">Pop Quiz 2 [AllToAll time]:</b> Using an 8xH100 node with 450 GB/s unidirectional bandwidth, how long does AllToAll<sub>X->N</sub>(bf16[B<sub>X</sub>, N]) take? What if we know only 4 of 8 entries will be non-zero?

{% details Click here for the answer. %}

**Answer:** From the above, we know that in the dense case, the cost is $B \cdot (N-1) / (W \cdot N^2)$, or $B / (W \cdot N)$. If we know only $\frac{1}{2}$ the entries will be non-padding, we can send $B \cdot k/N / (W \cdot N) = B / (2 \cdot W \cdot N)$, roughly half the overall cost.

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaway:** The cost of an AllToAll on an array of $B$ bytes on GPU within a single node is about $T_\text{comms} = (B \cdot (8 - 1)) / (8^2 \cdot W_\text{GPU egress}) \approx B / (8 \cdot W_\text{GPU egress})$. For a ragged (top-$k$) AllToAll, this is decreased further to $(B \cdot k) / (64 \cdot W_\text{GPU egress})$.</p>

**Empirical measurements:** here is an empirical measurement of AllReduce bandwidth over an 8xH100 node. The Algo BW is the measured bandwidth (bytes / runtime) and the Bus BW is calculated as $2 \cdot W \cdot (8 - 1) / 8$, theoretically a measure of the actual link bandwidth. You’ll notice that we do achieve close to 370GB/s, less than 450GB/s but reasonably close, although only around 10GB/device. This means although these estimates are theoretically correct, it takes a large message to realize it.

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>Figure:</b> AllReduce throughput for an 8xH100 node with SHARP disabled. The blue curve is the empirical link bandwidth, calculated as $2 * \text{bytes} * (N - 1) / (N * \text{runtime})$ from the empirical measurements. Note that we do not get particularly close to the claimed bandwidth of 450GB/s, even with massive 10GB arrays." %}

This is a real problem, since it meaningfully complicates any theoretical claims we can make, since e.g. even an AllReduce over a reasonable sized array, like LLaMA-3 70B’s MLPs (of size `bf16[8192, 28672]`, or with 8-way model sharding, `bf16[8192, 3584] = 58MB`) can achieve only around 150GB/s compared to the peak 450GB/s. By comparison, TPUs achieve peak bandwidth at much lower message sizes (see Appendix B).

<p markdown=1 class="takeaway">**Takeaway:** although NVIDIA claims bandwidths of about 450GB/s over an H100 NVLink, it is difficult in practice to exceed 370 GB/s, so adjust the above estimates accordingly.</p>

**In network reductions:** Since the Hopper generation, NVIDIA switches have supported ["SHARP" (Scalable Hierarchical Aggregation and Reduction Protocol)](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/) which allows for "in-network reductions". This means *the network switches themselves* can do reduction operations and multiplex or "MultiCast" the result to multiple target GPUs:

{% include figure.liquid path="assets/gpu/sharp-algorithm.png" class="img-fluid" caption="<b>Figure:</b> an AllReduce without SHARP has 2x the theoretical cost because it has to pass through each GPU twice. In practice, speedups are only about 30% (from NCCL 2.27.5)." %}

Theoretically, this close to halves the cost of an AllReduce, since it means each GPU can send its data to a top-level switch which itself performs the reduction and broadcasts the result to each GPU without having to egress each GPU twice, while also reducing network latency.

$$T_\text{SHARP AR comms} = \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

Note that this is exact and not off by a factor of $1/N$, since each GPU egresses $B \cdot (N - 1) / N$ first, then receives the partially reduced version of its local shard (ingress of $B/N$), finishes the reductions, then egresses $B/N$ again, then ingresses the fully reduced result (ingress of $B \cdot (N - 1) / N$), resulting in exactly $B$ bytes ingressed.

However, in practice we see about a 30% increase in bandwidth with SHARP enabled, compared to the predicted 75%. This gets us up merely to about 480GB/s effective collective bandwidth, not nearly 2x.

{% include figure.liquid path="assets/gpu/sharp-all-reduce-cost.png" class="img-fluid" caption="<b>Figure:</b> empirical measurements of AllReduce algo bandwidth with and without NVIDIA SHARP enabled within a node. The gains amount to about 30% throughput improvement at peak, even though algorithmically it ought to be able to achieve closer to a 75% gain." %}

<p markdown=1 class="takeaway">**Takeaway:** in theory, NVIDIA SHARP (available on most NVIDIA switches) should reduce the cost of an AllReduce on $B$ bytes from about $2 * B / W$ to $B / W$. However, in practice we only see a roughly 30% improvement in bandwidth. Since pure AllReduces are fairly rare in LLMs, this is not especially useful.</p>

### Cross-node collectives

When we go beyond the node-level, the cost is a bit more subtle. When doing a reduction over a tree, you can think of reducing from the bottom up, first within a node, then at the leaf level, and then at the spine level, using the normal algorithm at each level. For an AllReduce especially, you can see that this allows us to communicate less data overall, since after we AllReduce at the node level, we only have to egress $B$ bytes up to the leaf instead of $B * N$.

**How costly is this?** To a first approximation, because we have full bisection bandwidth, the cost of an AllGather or ReduceScatter is roughly the buffer size in bytes divided by the node egress bandwidth (400GB/s on H100) *regardless of any of the details of the tree reduction.*

$$T_\text{AG or RS comms} = \frac{\text{bytes}}{W_\text{node egress}} \underset{H100}{=} \frac{\text{bytes}}{\text{400e9}}$$

where $W_\text{node}$ egress is generally 400GB/s for the above H100 network (8x400Gbps IB links egressing each node). The cleanest way to picture this is to imagine doing a ring reduction over *every node in the cluster*. Because of the fat tree topology, we can always construct a ring with $W_\text{node}$ egress between any two nodes and do a normal reduction. The node-level reduction will (almost) never be the bottleneck because it has a higher overall bandwidth and better latency, although in general the cost is

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network}) = \max\left[\frac{\text{bytes}}{W_\text{GPU egress}}, \frac{\text{bytes}}{W_\text{node egress}}\right]$$

{% details You can see a more precise derivation here. %}

We can be more precise in noting that we are effectively doing a ring reduction at each layer in the network, which we can mostly overlap, so we have:

$$T_\text{AG or RS comms} = \text{bytes} \cdot max_\text{depth i}\left[\frac{D_i - 1}{D_i \cdot W_\text{link i}}\right]$$

where $D_i$ is the degree at depth $i$ (the number of children at depth $i$), $W_\text{link i}$ is the bandwidth of the link connecting each child to node $i$.

Using this, we can calculate the available AllGather/AllReduce bandwidth as $min_\text{depth i}(D_i * W_\text{link i} / (D_i - 1))$ for a given topology. In the case above, we have:

* **Node:** $D_\text{node}$ = 8 since we have 8 GPUs in a node with Wlink i = 450GB/s. Thus we have an AG bandwidth of `450e9 * 8 / (8 - 1) = 514GB/s`.
* **Leaf:** $D_\text{leaf}$ = 32 since we have 32 nodes in an SU with Wlink i = 400GB/s (8x400Gbps IB links). Thus our bandwidth is `400e9 * 32 / (32 - 1) = 413GB/s`.
* **Spine:** $D_\text{spine}$ = 4 since we have 4 SUs with $W_\text{link i}$ = 12.8TB/s (from `8 * 16 * 2 * 400Gbps` links above). Our bandwidth is `12.8e12 * 4 / (4 - 1) = 17.1TB/s`.

Hence our overall AG or RS bandwidth is `min(514GB/s, 413GB/s, 17.1TB/s) = 413GB/s` at the leaf level, so in practice $T_\text{AG or RS comms} = B / \text{413GB/s}$, i.e. we have about 413GB/s of AllReduce bandwidth even at the highest level. For an AllReduce with SHARP, it will be slightly lower than this (around 400GB/s) because we don’t have the $(N - 1) / N$ factor. Still, 450GB/s and 400GB/s are close enough to use as approximations.

{% enddetails %}

**Other collectives:** AllReduces are still 2x the above cost unless SHARP is enabled. NVIDIA sells SHARP-enabled IB switches as well, although not all providers have them. AllToAlls do change quite a bit cross-node, since they aren't "hierarchical" in the way AllReduces are. If we want to send data from every GPU to every other GPU, we can't use take advantage of the full bisection bandwidth at the node level. That means if we have an N-way AllToAll that spans $M = N / 8$ nodes, the cost is

$$T_\text{AllToAll comms} = \frac{B \cdot (M - 1)}{M^2 \cdot W_\text{node egress}} \approxeq \frac{B}{M \cdot W_\text{node egress}}$$

which effectively has 50GB/s rather than 400GB/s of bandwidth. We go from $B / (8 * \text{450e9})$ within a single H100 node to $B / (2 \cdot \text{400e9})$ when spanning 2 nodes, a more than 4x degradation.

Here is a summary of the 1024-GPU DGX H100 SuperPod architecture:

|   Level   | Number of GPUs | Degree (# Children) | Switch Bandwidth (full-duplex, TB/s) | Cable Bandwidth (full-duplex, TB/s) | Collective Bandwidth (GB/s) |
| :-------: | :------------: | :-----------------: | :----------------------------------: | :---------------------------------: | :-------------------------: |
|   Node    |       8        |          8          |                 6.4                  |                 3.6                 |             450             |
| Leaf (SU) |      256       |         32          |                 25.6                 |                12.8                 |             400             |
|   Spine   |      1024      |          4          |                 51.2                 |                51.2                 |             400             |

We use the term "Collective Bandwidth" to describe the effective bandwidth at which we can egress either the GPU or the node. It’s also the $\text{bisection bandwidth} * 2 / N$.

<p markdown=1 class="takeaway">**Takeaway:** beyond the node level, the cost of an AllGather or ReduceScatter on B bytes is roughly $B / W_\text{node egress}$, which is $B / \text{400e9}$ on an H100 DGX SuperPod, while AllReduces cost twice as much unless SHARP is enabled. The overall topology is a fat tree designed to give constant bandwidth between any two pairs of nodes.</p>

**Reductions when array is sharded over a separate axis:** Consider the cost of a reduction like

$$\text{AllReduce}_X(A[I_Y, J]\ \{ U_X \})$$

where we are AllReducing over an array that is itself sharded along another axis $Y$. On TPUs, the overall cost of this operation is reduced by a factor of $1 / Y$ compared to the unsharded version since we’re sending $1 / Y$ as much data per axis. On GPUs, the cost depends on which axis is the "inner" one (intra-node vs. inter-node) and whether each shard spans more than a single node. Assuming $Y$ is the inner axis, and the array has $\text{bytes}$ total bytes, the overall cost is reduced effectively by $Y$, but only if $Y$ spans multiple nodes:

$$T_\text{comms at node} = \frac{\text{bytes}}{W_\text{GPU egress}} \cdot \frac{1}{\min(Y, D_\text{node})}$$

$$T_\text{comms in scale-out network} = \frac{\text{bytes}}{W_\text{node egress}} \cdot \frac{D_\text{node}}{\max(D_\text{node}, Y)}$$

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network})$$

where N is the number of GPUs and again $D_\text{node}$ is the number of GPUs in a node (the degree of the node). As you can see, if $Y < D_\text{node}$, we get a win at the node level but generally don’t see a reduction in overall runtime, while if $Y > D_\text{node}$, we get a speedup proportional to the number of nodes spanned.

If we want to be precise about the ring reduction, the general rule for a tree AllGather<sub>X</sub>(A<sub>Y</sub> { U<sub>X</sub> }) (assuming Y is the inner axis) is

$$T_\text{AR or RS comms} = \text{bytes} \cdot \max_{\text{depth } i}\left[\frac{D_i - 1}{D_i \cdot \max(Y, S_{i-1}) \cdot W_{\text{link } i}}\right]$$

where $S_i$ is M * N * …, the size of the subnodes below level i in the tree. This is roughly saying that the more GPUs or nodes we span, the greater our available bandwidth is, but only within that node.

**Pop Quiz 3 [Sharding along 2 axes]:** Say we want to perform $\text{AllGather}_X(\text{bf16}[D_X, F_Y])$ where $Y$ is the inner axis over a single SU (256 chips). How long will this take as a function of $D$, $F$, and $Y$?

{% details Click here for the answer. %}

**Answer:** We can break this into two cases, where Y <= 8 and when Y > 8. When $Y <= 8$, we remain bounded by the leaf switch, so the answer is, as usual, $T_\text{comms} = 2 * D * F * (32 - 1) / (32 * 400e9)$. When Y > 8, we have from above, roughly

$$T_\text{comms} = \frac{2 \cdot D \cdot F \cdot 256}{Y \cdot \text{12.8e12}} = \frac{2DF}{Y \cdot \text{50GB/s}}$$

For `D = 8192`, `F = 32,768`, we have:

{% include figure.liquid path="assets/gpu/sharded-all-gather-cost.png" class="img-fluid" caption="<b>Figure:</b> theoretical cost of a sharded AllGather as the inner axis spans more nodes." %}

Note how, if we do exactly 8-way model parallelism, we do in fact reduce the cost of the node-level reduction by 8 but leave the overall cost the same, so it’s free but not helpful in improving overall bandwidth.

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaway:** when we have multiple axes of sharding, the cost of the outer reduction is reduced by a factor of the number of nodes spanned by the inner axis.</p>

### Quiz 4: Collectives

**Question 1 [SU AllGather]:** Consider only a single SU with M nodes and N GPUs per node. Precisely how many bytes are ingressed and egressed by the node level switch during an AllGather? What about the top-level switch?

{% details Click here for the answer. %}

**Answer:** Let’s do this step-by-step, working through the components of the reduction:

1. Each GPU sends $B / MN$ bytes to the switch, for a total ingress of $NB / MN = B / M$ bytes ingress.
2. We egress the full $B / M$ bytes up to the spine switch.
3. We ingress $B * (M - 1) / M$ bytes from the spine switch
4. We egress $B - B / MN$ bytes $N$ times, for a total of $N * (B - B / MN) = NB - B / M$.

The total is $B$ ingress and $BN$ egress, so we should be bottlenecked by egress, and the total time would be $T_\text{AllGather} = BN / W_\text{node} = B / \text{450e9}$.

For the spine switch, the math is actually simpler. We must have $B / M$ bytes ingressed M times (for a total of $B$ bytes), and then $B (M - 1) / M$ egressed $M$ times, for a total of $B * (M - 1)$ out. Since this is significantly larger, the cost is $T_\text{AllGather} = B \cdot (M - 1) / (M \cdot W_\text{node}) = B \cdot (M - 1) / (M \cdot \text{400e9})$.

{% enddetails %}

**Question 2 [Single-node SHARP AR]:** Consider a single node with N GPUs per node. Precisely how many bytes are ingressed and egressed by the switch during an AllReduce using SHARP (in-network reductions)?

{% details Click here for the answer. %}

**Answer:** As before, let’s do this step-by-step.

1. Each GPU sends $B * (N - 1) / N$ bytes, so we have $N * B * (N - 1) / N = B * (N - 1)$ ingressed.
2. We accumulate the partial sums, and we send back $B / N$ bytes to each GPU, so $N * B / N = B$ bytes egressed.
3. We do a partial sum on the residuals locally, then send this back to the switch. This is a total of $N * B / N = B$ bytes ingressed.
4. We capture all the shards and multicast them, sending $B * (N - 1) / N$ to $N$ destinations, for a total of $B * (N - 1) / N * N = B * (N - 1)$ egressed.

Therefore the total is $B * (N - 1) + B = BN$ bytes ingressed and egressed. This supports the overall throughput being exactly $B / W_\text{egress}$.

{% enddetails %}

**Question 3 [Cross-node SHARP AR]:** Consider an array bf16[D<sub>X</sub>, F<sub>Y</sub>] sharded over a single node of N GPUs. How long does AllReduce(bf16[D, F<sub>Y</sub>] { U<sub>X</sub> }) take? You can assume we do in-network reductions. Explain how this differs if we have more than a single node?

{% details Click here for the answer. %}

**Answer:** We can try to modify the answer to the previous question above. Basically, we first egress $B * (X - 1) / XY$ bytes from each GPU, then send back $B / XY$ to each GPU, then send that same amount back to the switch, then send $B * (X - 1) / XY$ back to each GPU. The total is $NB / Y$ ingress and egress, so the total time is $T_\text{comms} = NB / (Y * N * W_\text{link}) = N * 2DF / (Y * N * W_\text{link}) = 2 * D * F / (Y * W_\text{link})$, so the total time does decrease with $Y$.

If we go beyond a single node, we can do roughly the same reduction as above, but when we egress the node-level switch, we need to send all B bytes, not just $B / Y$. This is because we need to keep each shard separate.

{% enddetails %}

**Question 4 [Spine level AR cost]:** Consider the same setting as above, but with $Y = 256$ (so the AR happens at the spine level). How long does the AllReduce take? Again, feel free to assume in-network reductions.

{% details Click here for the answer. %}

**Answer:** This lets us take advantage of the rather ludicrous amount of bandwidth at the spine level. We have 25.6TB/s of bandwidth over 4 nodes, so an AllReduce bandwidth of 6.4TB/s. Using SHARP, this could take as little as `2 * D * F / 6.4e12` seconds.

{% enddetails %}

**Question 5 [2-way AllGather cost]:** Calculate the precide cost of an AllGather of $B$ bytes over exactly 2 nodes. *Make sure to calculate the precise cost and not the approximation, and consider both the intra-node and cross-node cost.*

{% details Click here for the answer. %}

**Answer:** At the node level, we have $T_\text{comms} = B * 7 / (8 * \text{450e9}) = B / \text{514e9}$ while beyond we actually have $T_\text{comms} = B * (2 - 1) / (2 * \text{400e9}) = B / \text{800e9}$. Thus, we’re actually bounded by the node level reduction and not the leaf level! This motivates e.g. DeepSeek v3 which does 2-way Data Parallelism.

{% enddetails %}

## Rooflines for LLM Scaling on GPUs

Now let’s look at what this has all been building towards: understanding rooflines for LLM scaling on GPU. This is to complement the TPU training chapter [here](../training). As we did there, the goal here is to look at the total $T_\text{math}$ and $T_\text{comms}$ for different parallelism strategies and understand at what point $T_\text{comms} > T_\text{math}$. As before, we consider only the MLP block with operations

$$\text{MLP}(x) \equiv x[B, D] *_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D]$$

where $B$ is the global batch size **in tokens** (i.e. $B = \text{batch size} \cdot \text{sequence length}$).

Here we'll reproduce the table above showing effective bandwidths at both the GPU and node level:

|  Node Type  | GPUs per node | GPU egress bandwidth | Node egress bandwidth |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

**Note:** Both the GPU and node egress bandwidths determine rooflines for our LLMs. We'll use the term $W_\text{collective}$ to describe either the GPU or node bandwidths depending on whether we are operating within or above the node level.

Let’s look at the compute communication rooflines as we did for TPUs for **data parallelism, tensor parallelism, pipeline parallelism, expert parallelism,** and combinations thereof. For the rest of this section we'll focus on H100 rooflines for specific calculations. GB200-NVL72 has the same general rooflines but because we have a larger node egress bandwidth, we can sometimes be bottlenecked at the node level instead.

### Data Parallelism

As noted before, DP and ZeRO sharding involve either a weight AllReduce or a ReduceScatter + AllGather in the backward pass. Since these both have the same cost, to be compute-bound for pure data parallelism or FSDP *without in-network reductions*, we have, per layer, in the backward pass, with an axis of size X:

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{W_\text{collective}}$$

Therefore, for $T_\text{math} > T_\text{comms}$, we need $B / (XC) > 1 / W_\text{collective}$ or 

$$\frac{B}{X} > \frac{C}{W_\text{collective}}$$

where $W_\text{collective}$ is either the GPU or node level egress bandwidth depending on whether we're sharding within a node or across nodes. Thus:

* **Within a node**, we just need the per-GPU **token** batch size > $\text{990e12} / \text{450e9} = 2200$.
* **Within an SU or at the spine level**, BS > $\text{990e12} / \text{400e9} = 2475$.

This is quite a bit higher than on a TPU, where the number is 850 with all three axes. For instance, LLaMA-3, which trained on 16000 H100s would need a batch size of at least 40M tokens (for reference, they used 16M). DeepSeek v3 trained on 2048 H800 GPUs with lower 300GB/s of bandwidth (instead of 450GB/s on H100) would need $\text{990e12} / \text{300e9} = 3300$ tokens per GPU, or about 6.7M (in practice, they used 4M).

With in-network reductions enabled and using pure data parallelism, theoretically we have 2x the AllReduce bandwidth, which would halve both of these numbers. However, in practice the benefit is closer to 30%, which only really makes up for the fact that we typically struggle to reach the reported numbers. Furthermore, because pure data parallelism is rarely useful, this basically doesn’t matter in practice.

**MoE models:** For a Mixture of Experts (MoE) model, where we have E experts and k experts per token, this increases to

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot EDF}{W_\text{collective}}$$

which inflates the per-GPU token batch size by a factor of $E/k$, i.e.

$$\frac{B}{X} > \frac{E}{k} \frac{C}{W_\text{collective}}$$

For example, the new OpenAI OSS model with $k=4$ and $E=128$, this increases to `32 * 2475  = 79,200` across nodes, a kind of ridiculously high number.

**What happens when X is small?** When we do only e.g. 2-node data parallelism, we benefit from the $(X - 1) / X$ scaling, which gives us

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N * C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF \cdot (X-1)}{X \cdot W_\text{collective}}$$

where X is the number of nodes and $N = 8 \cdot X$. Then for a dense model we have $B / N > \alpha \cdot (X - 1) / X$, or e.g. $B / N > \text{1237}$, half the above value. You’ll notice 2-way data parallelism fairly often for this reason.

<p markdown=1 class="takeaway">**Takeaway:** Data parallelism and ZeRO sharding require a per-GPU batch size of about 2500 tokens to be compute-bound on an H100 or B200, assuming perfect overlap and FLOPs utilization. For MoE models, this increases by a factor of $E / k$, the ratio of total to activated parameters. When doing a small amount of data parallelism, the critical batch size decreases.</p>

### Tensor Parallelism

Tensor parallelism requires an AllGather and ReduceScatter over the activations, which we need to overlap with the MLP FLOPs. In other words, in the forward pass, we have

$$T_\text{math} = \frac{2\cdot 2 \cdot BDF}{Y \cdot C}$$

$$T_\text{comms} = \frac{2\cdot 2 \cdot BD}{W_\text{collective}}$$

which to be compute-bound gives us the rule 

$$Y < \frac{F \cdot W_\text{collective}}{C}$$

Within a node, this gives us about $F / 2200$ or $F / 2475$ beyond a node. For $F=\text{28000}$ like LLaMA-3, this is about 11-way TP (or rounding down, about 8-way, which is how large a node is). As with above, we get an extra 2X bandwidth when we span exactly 2 nodes, so we can generally do 16-way data parallelism ($F > 2475 \cdot (Y - 8)$), which gives us up to 19-way model parallelism in theory.

<p markdown=1 class="takeaway">**Takeaway:** Tensor parallelism over an axis of size Y with feed-forward dimension F becomes communication-bound when the $Y > F / 2475$, which generally constrains us to only intra-node TP or at most 2-node TP.</p>

### Expert Parallelism

As we’ve already noted above, Mixture of Expert (MoE) models come with E times more model weights with only k times more FLOPs, making data parallelism significantly harder. We can mitigate this somewhat by sharding the our weights along the expert dimension, i.e. W<sub>in</sub>[E<sub>Z</sub>, D, F]. To do the MLP block, we need to introduce 2x AllToAll to send our activations to the corresponding experts.

As noted above, the cost of this AllToAll<sub>Z->k</sub>([B, D, k]) if it spans multiple nodes is roughly $T_\text{AllToAll} = 2 \cdot B \cdot D \cdot (Z-8)/Z \min(8 * k / Z, 1)$, so for pure expert parallelism we need

$$T_\text{math} = \frac{4 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}$$

$$T_\text{comms} = \frac{4 \cdot B \cdot D \cdot (Z-8)}{W \cdot Z} \cdot \min\left(\frac{8 \cdot k}{Z}, 1\right)$$

We either need $K > Z/8$ with $F > \alpha \cdot (Z - 8)/k$ or $Z \gg K$ and $F > 8 \cdot \alpha$, where $\alpha = C/W$. This gives you two domains in which expert parallelism is possible, one with a small amount of expert parallelism (roughly 2-node) and small $F$, or one with large $F$ and $Z$ arbitrarily large (up to E-way expert parallelism).

You’ll see both cases in practice, either a small amount of expert-parallelism (like DeepSeek v3 which has very small F and relatively small, restricted cross-node expert parallelism), or models with large F, in which case we can do significant cross-node EP alongside TP.

<p markdown=1 class="takeaway">**Takeaway:** if $F < 8 * C / W_\text{node}$, expert parallelism can span 1-2 nodes with similar (slightly lower) cost to TP, or if $F > 8 * C / W_\text{node}$, we can do a significant amount of expert parallelism (up to $E$ nodes) with relatively low cost.</p>

### Pipeline Parallelism

Pipeline parallelism splits layers across nodes with an extremely low communication cost, since we are just sending small microbatches of activations every couple layers. Historically pipelining has suffered from "pipeline bubbles", but with new zero-bubble pipelining approaches, it is typically possible to do without.

The overall communication cost of pipelining is tiny: with $N_\text{MB}$ microbatches and $N_\text{stages}$, we have $T_\text{comms per hop} = 2 \cdot B \cdot D / (W \cdot N_\text{MB})$ and $N_\text{MB} + N_\text{stages} - 2$ hops, so roughly

$$T_\text{total PP comms} = \frac{2BD}{W \cdot N_\text{MB}} \cdot (N_\text{MB} + N_\text{stages} - 2)$$

$$T_\text{per-layer comms} \approx 1.5 \cdot \frac{2BD}{W \cdot N_\text{layers}}$$

Since we are dividing by $N_\text{layers}$, this is vastly smaller than any of the other costs. In other words, from a communication standpoint, pipelining is basically free. So why don’t we just do pipelining? There are a few reasons:

(1) **Code complexity:** pipelining doesn’t fit nicely as nicely into automatic parallelism frameworks (like XLA’s GSPMD) as other approaches. Because it introduces microbatching to hide pipeline bubbles, it changes the structure of the program, and custom zero-bubble pipeline schedules exacerbate this problem by requiring complicated interleaving of the forward and backward pass.

(2) **Pipelining makes data parallelism and FSDP hard:** probably the biggest reason not to do pipelining is that it plays badly with FSDP and data parallelism. ZeRO-3 sharding in particular works badly, since it requires us to AllGather the weights on every microbatch which doesn’t work when we have only $B / N_\text{microbatches}$ tokens to amortize the AllGather cost. Furthermore, during the backward pass, *we can’t AllReduce or ReduceScatter the gradients until the last microbatch has passed a given stage, which means we have significant non-overlapped communication time.*

{% include figure.liquid path="assets/gpu/pipeline-bubble.png" class="img-fluid" caption="<b>Figure:</b> an example 2 stage, 2 microbatch pipeline. F denotes a stage forward pass and B is a stage backward pass (2x the cost). G denotes the data-parallel AllReduces, which can be significantly longer than the time of a single microbatch." %}

(3) **Pipeline bubbles and step imbalance:** As you can see in the (bad) pipeline schedule above, it is easy to have significant bubbles (meaning wasted compute) during a naive pipeline schedule. Above, the second stage is idle on step 0, the first stage is idle from step 2 to 3, and the second stage is again idle on the last step. While we can avoid these somewhat with careful scheduling, we still often have some bubbles. We also have to pass activations from one stage to the next on the critical path, which can add overhead:

{% include figure.liquid path="assets/gpu/pipeline-transfer.png" class="img-fluid" caption="<b>Figure:</b> an example pipeline showing transfer cost in red. This shifts stages relative to each other and increases the pipeline bubble overhead." %}

There are workarounds for each of these issues, but they tend to be complicated to implement and difficult to maintain, but pipelining remains a technique with low communication cost relative to other methods.

**Caveat about latency:** As noted before, GPUs struggle to achieve full AllReduce bandwidth even with fairly large messages. This means even if we in theory can scale e.g. expert-parallel AllToAlls across multiple nodes, we may struggle to achieve even 50% of the total bandwidth. This means we do try to keep TP or EP within a smaller number of nodes to minimize latency overhead.

### Examples

**What does DeepSeek do?** For reference, [DeepSeek V3](https://arxiv.org/abs/2412.19437) is trained with 2048 H800 GPUs with:

* 64-way Expert Parallelism (EP) spanning 8 nodes
* 16-way Pipeline Parallelism (PP)
* 2-way ZeRO-1 Data Parallelism (DP)

They had a steady state batch size of `4096 * 15360 = 62,914,560` tokens, or 30k tokens per GPU. You can see that this is already quite large, but their model is also very sparse (k=8, E=256) so you need a fairly large batch size. You can see that with 64-way EP and 16-way PP, we end up with 1024-way model parallelism in total, which means the AllReduce is done at the spine level, and because it’s only 2-way, we end up with $2 / (2 - 1) = 2$ times more bandwidth in practice. This also helps reduce the cost of the final data-parallel AllReduce overlapping with the final pipeline stages.

**What does LLaMA-3 do?** LLaMA-3 trains with a BS of 16M tokens on 16k GPUs, or about 1k tokens per GPU. They do:

* 8-way Tensor Parallelism within a node (TP)
* 16-way Pipeline Parallelism (PP)
* 128-way ZeRO-1 Data Parallelism

This is also a dense model so in general these things are pretty trivial. The 16-way PP reduces the cost of the data parallel AllReduce by 16x, which helps us reduce the critical batch size.

### TLDR of LLM Scaling on GPUs

Let’s step back and come up with a general summary of what we’ve learned so far:

* **Data parallelism or FSDP (ZeRO-1/3) requires a local batch size of about 2500 tokens per GPU**, although in theory in-network reductions + pure DP can reduce this somewhat.
* **Tensor parallelism is compute-bound up to about 8-ways** but we lack the bandwidth to scale much beyond this before becoming comms-bound. This mostly limits us to a single NVLink domain (i.e. single-node or need to use GB200NVL72 with to 72 GPUs).
* **Any form of model parallelism that spans multiple nodes can further reduce the cost of FSDP**, so we often want to mix PP + EP + TP to cross many nodes and reduce the FSDP cost.
* **Pipeline parallelism works well if you can handle the code complexity of zero-bubble pipelining and keep batch sizes fairly large to avoid data-parallel bottlenecks.** Pipelining usually makes ZeRO-3 impossible (since you would need to AllGather on each pipeline stage), but you can do ZeRO-1 instead.

**At a high level, this gives us a recipe for sharding large models on GPUs:**

* For relatively small dense models, aggressive FSDP works great if you have the batch size, possibly with some amount of pipelining or tensor parallelism if needed.
* For larger dense models, some combination of 1-2 node TP + many node PP + pure DP works well.
* For MoEs, the above rule applies but we can also do expert parallelism, which we prefer to TP generally. If $F > 8 * C / W_\text{node}$, we can do a ton of multi-node expert parallelism, but otherwise we’re limited to roughly 2-node EP.

### Quiz 5: LLM rooflines

**Question 1 [B200 rooflines]:** A B200 DGX SuperPod (**not GB200 NVL72**) has 2x the bandwidth within a node (900GB/s egress) but the same amount of bandwidth in the scale-out network (400GB/s) ([source](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/network-fabrics.html)). The total FLOPs are reported above. How does this change the model and data parallel rooflines?

{% details Click here for the answer. %}

**Answer:** Our FLOPs/s in bfloat16 increases from 990 to 2250 TFLOPs, a 2.25x increase. With 2x the bandwidth, within a node, our rooflines stay roughly the same. For TP, for example, the critical intensity goes up to `2250e12 / 900e9 = 2500`, so we have a limit of $Y < F / 2500$, only slightly higher (and this doesn’t help us unless the node size increases).

Beyond a node, however, the lack of additional bandwidth actually makes it even harder for us to be compute-bound! For instance, for data parallelism, our critical batch size increases to `2250e12 / 400e9 = 5625`, because our GPU can do significantly more FLOPs with the same bandwidth.

GB200 SuperPods with 72-GPU nodes change this by adding more egress bandwidth ([source](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)).

{% enddetails %}

**Question 2 [How to shard LLaMA-3 70B]:** Consider LLaMA-3 70B, training in bfloat16 with fp32 optimizer state with Adam.

1. At a minimum, how many H100s would we need simply to store the weights and optimizer?
2. Say we want to train on 4096 H100 GPUs for 15T tokens. Say we achieved 45% MFU (Model FLOPs Utilization). How long would it take to train?
3. LLaMA-3 70B has `F = 28,672` and was trained with a batch size of about 4M tokens. What is the most model parallelism we could do without being comms-bound? With this plus pure DP, could we train LLaMA-3 while staying compute-bound on 4k chips? What about ZeRO-3? What about with 8-way pipelining? *Note: consider both the communication cost and GPU memory usage.*

{% details Click here for the answer. %}

1. We need 2 bytes for the weights and 8 for the optimizer state, so at least 700GB. With 80GB of DRAM, we’ll need at least 9 GPUs at a minimum, or (rounding up) at least 2 8xH100 nodes. This would take forever to train and wouldn’t hold the gradient checkpoints, but it’s a lower bound.
2. This will require a total of `6 * 70e9 * 15e12 = 6.3e24 bf16 FLOPs`. Each GPU can do `990e12` FLOPs, so at 45% MFU we can do 1.8e18 FLOPs/s. Thus the whole thing will take 3.5e6 seconds, or 40 days.
3. Within a node, we have 450GB/s of bandwidth, so the limit is roughly `F / 1995 = 28672 / 1995 = 14.372`. Since this doesn’t span 2 nodes, it realistically means we’d go up to 8-way model parallelism.
   1. This would then require us to do 512 way DP. Firstly, we need to see if we have enough memory. Since our model is only sharded 8-ways, this would mean `700GB / 8 = 87.5GB / GPU`, which won’t fit, so no!
   2. With ZeRO-3 and 8-way TP, we’ll be doing 512-way ZeRO-3. This won’t have any issue with memory because we’re sharding everything aggressively. We’ll have a per-GPU batch size of `4e6 / 4096 = 976`. This is quite low, even below our pure DP limit, and this is twice that limit because we have to move our weights. So no.
   3. With 8-way pipelining, each model parallel shard now spans 8 nodes. As we’ve seen, this reduced the cost of our leaf-level AllGathers by 8, so the overall AllReduce/AllGather bandwidth there goes from 400GB/s to `8 * 400GB/s = 3200GB/s`. The roofline then is `990e12 / 3200e9 = 309`, so we should be good! We just need to implement pipelining efficiently.

{% enddetails %}

**Question 3 [Megatron-LM hyperparams]:** Consider this figure from the [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM) highlighting their high MFU numbers.

{% include figure.liquid path="assets/gpu/megatron-hparams.png" class="img-fluid" %}

Note that their sequence length is 4096 everywhere. For the 16B, 70B, and 314B models, what is the per-GPU token batch size? Assuming data parallelism is the outermost axis and assuming bfloat16 reductions, determine whether each of these is theoretically compute-bound or communication-bound, and whether there is a more optimal configuration available?

{% details Click here for the answer. %}

**Answer:** Let’s start with batch sizes per GPU.

* **16B**: `192 * 4096 / 192 = 4096` tokens per GPU
* **70B**: `384 * 4096 / 768 = 2048` tokens per GPU
* **314B**: `1536 * 4096 / 3072 = 2048` tokens per GPU

This means with the exception of the first, these all hover around 2k tokens per batch, which is notably around the critical threshold we calculated for FSDP. We had calculated that bound to be 2,472 tokens / GPU based on the spine level reduction, which should roughly come into play here. For both the 70B and 314B though, because we have 16 and 64-way model (PP + TP) sharding respectively, we get 2x and 8x better throughput at the spine level, which means we should be compute-bound at roughly 1k and 300 tokens / step respectively.

{% enddetails %}

## Acknowledgements and Further Reading

This chapter relied heavily on help from many knowledgeable GPU experts, including:

* Adam Paszke, who helped explain the realities of kernel programming on GPUs.
* Swapnil Patil, who first explained how GPU networking works.
* Stas Bekman, who pointed out that the empirical realities of GPUs are often different from the purported specs.
* Reiner Pope, who helped clarify how GPUs and TPUs compare at a hardware level.
* Frédéric Bastien, who gave detailed feedback on the chip-level story.
* Nouamane Tazi, whose experience with LLM training on GPUs helped improve the roofline section.
* Sanford Miller, who helped me understand how GPUs are networked and how NVIDIA’s specifications compare to what’s often deployed in the field.

There’s a great deal of good reading on GPUs, but some of my favorites include:

* [SemiAnalysis’ History of the NVIDIA Tensor Core](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/): a fantastic article describing how GPUs transformed from video game engines to ML accelerators.
* [SemiAnalysis’ Analysis of Blackwell Performance](https://semianalysis.com/2024/04/10/nvidia-blackwell-perf-tco-analysis/): worth reading to understand the next generation of NVIDIA GPUs.
* [H100 DGX SuperPod Reference](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf): dry but useful reading on how larger GPU clusters are networked. [Here](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576) is a similar document about the GB200 systems.
* [Hot Chips Talk about the NVLink Switch](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf): fun reading about NVLink and NCCL collectives, especially including in-network reductions.
* [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437): a good example of a large semi-open LLM training report, describing how they picked their sharding setup.
* [How to Optimize a CUDA Matmul](https://siboehm.com/articles/22/CUDA-MMM): a great blog describing how to implement an efficient matmul using CUDA Cores, with an eye towards cache coherence on GPU.
* [HuggingFace Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook): a guide to LLM parallelism on GPUs, which partly inspired this chapter.
* [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html): a more GPU and PyTorch-focused tutorial on LLM rooflines and performance engineering.
* [Cornell Understanding GPU Architecture site](https://cvw.cac.cornell.edu/gpu-architecture): a similar guide to this book, comparing GPU and CPU internals more specifically.

## Appendix A: How does this change with GB200?

Blackwell introduces a bunch of major networking changes, including NVLink 5 with twice the overall NVLink bandwidth (900GB/s). B200 still has 8-GPU nodes, just like H100s, but GB200 systems (which combine B200 GPUs with Grace CPUs) introduce much larger NVLink domain (72 GPUs in NVL72 and in theory up to 576). This bigger NVLink domain also effectively increases the node egress bandwidth, which reduces collective costs above the node level.

{% include figure.liquid path="assets/gpu/b200-node.png" class="img-small" caption="<b>Figure:</b> a diagram showing how a GB200 NVL72 unit is constructed, with 18 switches and 72 GPUs." %}

Within a node, this increased bandwidth (from 450GB/s to 900GB/s) doesn't make much of a difference because we also double the total FLOPs/s of each GPU. Our rooflines mostly stay the same, although because NVLink has much better bandwidth, Expert Parallelism becomes easier.

Beyond a node, things change more. Here's a SuperPod diagram from [here](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576).

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>Figure:</b> a diagram showing a GB200 DGX SuperPod of 576 GPUs." %}

As you can see, the per-node egress bandwidth increases to `4 * 18 * 400 / 8 = 3.6TB/s`, up from 400GB/s in H100. This improves the effective cross-node rooflines by about 4x since our FLOPs/chip also double. Now we may start to worry about whether we're bottlenecked at the node level rather than the scale-out level.

**Grace Hopper:** NVIDIA also sells GH200 and GB200 systems which pair some number of GPUs with a Grace CPU. For instance, a GH200 has 1 H200 and 1 Grace CPU, while a GB200 system has 2 B200s and 1 Grace CPU. An advantage of this system is that the CPU is connected to the GPUs using a full bandwidth NVLink connection (called NVLink C2C), so you have very high CPU to GPU bandwidth, useful for offloading parameters to host RAM. In other words, for any given GPU, the bandwidth to reach host memory is identical to reaching another GPU’s HBM.

## Appendix B: More networking details

Here’s a diagram of an NVLink 4 switch. There are 64 overall NVLink4 ports (each uses 2 physical lanes), and a large crossbar that handles inter-lane switching. TPUs by contrast use optical switches with mirrors that can be dynamically reconfigured.

{% include figure.liquid path="assets/gpu/nvlink4.png" class="img-fluid" caption="<b>Figure:</b> a lower level view of a single NVLink4 Switch." %}

At each level we can be bottlenecked by the available link bandwidth or the total switch bandwidth.

* **Node level:** at the node level, we have 4 * 1.6TB/s = 6.4TB/s of NVSwitch bandwidth, but each of our 8 GPUs can only egress 450GB/s into the switch, meaning we actually have a peak bandwidth of 450e9 * 8 = 3.6TB/s (full-duplex) within the node.
* **SU/leaf level:** at the SU level, we have 8 switches connecting 32 nodes in an all-to-all fashion with 1x400 Gbps Infiniband. This gives us 8 * 32 * 400 / 8 = 12.8TB/s of egress bandwidth from the nodes, and we have 8 * 1.6TB/s = 12.8TB/s at the switch level, so both agree precisely.
* **Spine level:** at the spine level, we have 16 switches connecting 32 leaf switches with 2x400 Gbps links, so we have 32 * 16 * 400 * 2 / 8 = 51.2TB/s of egress bandwidth. The 16 switches give us 16 * 1.6TB/s = 25.6TB/s of bandwidth, so this is the bottleneck at this level.

Per GPU, this gives us 450GB/s of GPU to GPU bandwidth at the node level, 50GB/s at the SU level, and 25 GB/s at the spine level.

**GPU empirical AR bandwidth:**

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>Figure:</b> AllReduce bandwidth on an 8xH100 cluster (intra-node, SHARP disabled)." %}

TPU v5p bandwidth (1 axis):

{% include figure.liquid path="assets/gpu/tpu-all-reduce-bw.png" class="img-fluid" caption="<b>Figure:</b> AllReduce bandwidth on a TPU v5p 4x4x4 cluster (along one axis)." %}

Here’s AllGather bandwidth as well:

{% include figure.liquid path="assets/gpu/gpu-all-gather-bw.png" class="img-fluid" caption="<b>Figure:</b> AllGather bandwidth on an 8xH100 cluster (intra-node)." %}

{% include figure.liquid path="assets/gpu/tpu-all-gather-bw.png" class="img-fluid" caption="<b>Figure:</b> AllGather bandwidth on a TPU v5e 8x16 cluster (along one axis)." %}

**More on AllToAll costs:**

Here we can compare the approximation $\min(K / Z) * (Z - 1) / Z$ to the true value of $(1 - ((Z - 1) / Z) ** K) * (Z - 1) / Z$. They’re similar except for small values of $Z$.

{% include figure.liquid path="assets/gpu/all-to-all-approx.png" class="img-fluid" caption="<b>Зураг:</b> Ragged AllToAll-ийн ойролцоо болон жинхэнэ зардлыг хэсгүүдийн тоо нэмэгдэхэд харьцуулсан байдал." %}