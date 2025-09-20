—
Энэхүү орчуулга нь MIT лицензийн дагуу эх бүтээлээс хөрвүүлэв.
Эх сурвалж: Austin et al., "How to Scale Your Model" (https://jax-ml.github.io/scaling-book/)
Орч.: Mongolian (mn)
—

---
layout: distill
title: "TPU дээр LLaMA 3-г сургах"
# permalink: /main/
description: "Өмнөх хэсэгт сурсан зүйлээ ашиглан TPU v5p дээр LLaMA 3 загваруудыг хэрхэн сургахыг нарийвчлан харцгаая. Эдгээр загварууд хэр том вэ? Янз бүрийн тохиргоонд сургах нь хэр үнэтэй вэ? Эдгээрийг хэрхэн хуваадаг вэ? Өмнөх хэсгүүдийг бодит загвар дээр хэрхэн хэрэгжүүлэхийг энгийн тооцооллоор харцгаая."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 6

previous_section_url: "../training"
previous_section_name: "5-р хэсэг: Сургалт"

next_section_url: ../inference
next_section_name: "7-р хэсэг: Дүгнэлт"

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
#   - Агуулгын жагсаалтын нэрүүд нь тухайн хэсгийн нэртэй яг ижил байх ёстой
#     ингэснээр бичлэг доторх холбоосууд зөв ажиллах болно.
#   - Гар аргаар markdown агуулгын жагсаалт хийхийн оронд энэ форматыг ашиглана уу.
toc:
  - name: "LLaMA 3 ямар харагддаг вэ?"
  - name: "Параметр ба FLOPs тоолох"
  - name: "LLaMA 3-70B-г сургалтанд хэрхэн хуваах вэ"
  - name: "Бодлогын жишээ"

# Доор нэмэлт постод зориулсан тусгай загвар (styles) хэрхэн оруулахыг харуулсан жишээ байна.
# Энэ нь энэ постын 'Layouts' хэсэгт ашиглагддаг.
# Хэрвээ та энэ постыг загвар (template) болгон ашиглах бол энэ _styles блокийг устгаарай.
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

_Энэ хэсгийн зорилго бол өмнөх хэсгээс авсан үр дүнг маш практик асуудалд хэрэглэх явдал юм: LLaMA 3 загварын бүлгийг (herd) сургах. Өмнөх хэсгүүдээс ялгаатай нь энэ удаа бид ихэнх ажлыг өөрөө хийхийг хүсэж байна. Энэ шалтгааны улмаас бид хариултуудыг нуусан байгаа, ингэснээр та эхлээд өөрөө хариулахыг оролдоорой. Үзэг аваад гараар бодоод үзээрэй!_

### LLaMA 3 ямар харагддаг вэ?

LLaMA-3 загварын гэр бүл<d-cite key="llama3"></d-cite> нь 3 үндсэн загвартай: LLaMA 3 8B, 70B, болон 405B. Бид голчлон 70B загварт анхаарна, харин 8B болон 405B загварыг та нар сүүлийн хэсгийн бодлогын хэсэгт өөрсдөө судална. Энд LLaMA 3-70B загварын архитектур байна. Энэ нь LLaMA-гийн [HuggingFace хуудаснаас](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json) авсан болно.

| **hyperparam**              | **утга** |
| --------------------------- | --------- |
| $$n_\text{layers}$$ (L)     | 80        |
| $$d_\text{model}$$ (D)      | 8,192     |
| $$d_{ff}$$ (F)              | 28,672    |
| $$n_\text{heads}$$ (N)      | 64        |
| $$n_\text{kv_heads}$$ (K)   | 8         |
| $$d_\text{qkv}$$ (H)        | 128       |
| $$n_\text{embeddings}$$ (V) | 128,256   |

To highlight how easy this is to find, here's the config itself, along with a mapping:

{% include figure.liquid path="assets/img/llama-json.png" class="img-fluid" %}

_It's useful to make a big table with these numbers for many different open-source LLMs, so you can quickly compare the design decisions they've made._

### Counting parameters and FLOPs

**Question:** From this table, can we calculate the LLaMA 3-70B parameter count? 🤫 Let's apply the content of [Section 4](../transformers) and see if we can get 70B!

| param            | formula                                                                                                                                           | count                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| FFW params       | d_model * d_ff * 3 (for gelu + out-projection) * n_layers                                                                                         | 8,192 * 8,192 * 3.5 * 3 * 80 = **56.3e9**                    |
| Vocab params     | 2 (input and output embeddings) * n_embeddings * d_model                                                                                          | 2 * 128,256 * 8,192 = **2.1e9**                              |
| Attention params | n_layers * [ 2 (for q embedding and concatenated output projection) * d_model * n_heads * d_qkv + 2 (for k and v) * d_model * n_kv_heads * d_qkv] | 80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = **12e9** |
|                  |                                                                                                                                                   | 56.3e9 + 2.1e9 + 12e9 = **70.4e9**                           |

That's great! We get the number we expect. You'll notice as expected that the FFW parameters totally dominate the overall parameter count, although attention is non-trivial.

<p markdown=1 class="takeaway">**Takeaway**: The 3 big weight matrices in the MLP block are so much larger than all the other arrays in the Transformer that we can typically almost ignore all other parameters when reasoning about model memory or FLOPs. For LLaMA 3-70B, they represent 56B of 70B parameters.</p>

Let's look at FLOPs now! *Remember the general rules for training from [Section 4](../transformers).*

**Question:** How many FLOPs does LLaMA-3 perform per token per training step? _This helps us determine how expensive the whole training process will be._

{% details Click here for the answer, once you've thought about it! %}

**Answer**: As shown in [Section 4](../transformers), we do roughly $$6 \cdot \text{param count}$$ FLOPs per token, so here that's roughly `6 * 70e9 = 4.2e11` FLOPs / token. That's about half a TFLOP per token per step. Assuming we're compute-bound, this should take roughly `4.2e11 / 4.59E+14 = 1ms` on a single TPU v5p chip, assuming perfect FLOPs utilization.

{% enddetails %}

**Question:** LLaMA 3 was trained for about 15 trillion tokens. How many FLOPs is that total?

{% details Click here for the answer, once you've thought about it! %}

**Answer**: That's easy, it's just `4.2e11 * 15e12 = 6.3e24 FLOPs` total. 6.3 yottaFLOPs. That's a lot! On a single TPU this would take `6.3e24 / 4.59E+14 = 435 years`. That's also a lot!

{% enddetails %}

**Question:** Let's say we wanted to train on a full TPU v5p pod with 16x20x28 = 8960 chips. How long would this take to train at 40% MFU in bfloat16, assuming we are compute-bound?

{% details Click here for the answer, once you've thought about it! %}

**Answer**: We know that each TPU v5p can perform 4.59e14 FLOPs / second. At 40% MFU, this will take about `T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6 seconds`. **This is about 44 days!** That's fairly reasonable, assuming we can actually achieve 40% MFU.

{% enddetails %}

**Question:** LLaMA 3-70B was pretrained with a batch size of about 4M tokens. How many TPUs do we need at minimum to train with this batch size? _You can assume bfloat16 parameters and float32 optimizer state, and that you checkpoint gradients 4 times per layer._

{% details Click here for the answer, once you've thought about it! %}

**Answer**: This question is primarily asking about memory usage, since that's the only strict constraint on available compute. During training, we have three primary uses of HBM: model parameters, optimizer state, and gradient checkpoints. If we assume bfloat16 weights, float32 optimizer state, and a _very_ conservative gradient checkpointing scheme (4 times per layer), we have:

| **Params** | 2 * 70GB | ~140GB |
| **Optimizer State** | 8 * 70GB | ~560GB |
| **Gradient Checkpoints** | 2 * 8192 * 4e6 * 4 * 80 | ~20.9TB |
| **Total**                |                         | ~21.6TB |

The total here is about 21.6TB. You notice that gradient checkpointing strongly dominates the memory picture, even with a very conservative checkpointing scheme. We could technically go to 1 checkpoint per layer, or do microbatching, but this is a reasonable picture. With these assumptions, since each TPU v5p has 96GB of HBM, we need `21.6e12 / 96e9 = 225` TPUs. That's not very much actually!

*Why wouldn't we do this?* Well, because it would take us `44 days * 8960 / 225 = 1752 days` to train. That's nearly four years. **That's a lot.** Still, this makes it clear that we're using these large clusters not because we're bound by memory but rather because we need the extra FLOPs.

{% enddetails %}

**Question:** Under the same assumptions as the question above, if we use 8960 TPU v5p chips, how much memory will we use per-chip?

{% details Click here for the answer, once you've thought about it! %}

**Answer**: Our total memory is still about 21.6TB, so per-chip we'll be using about 2.4GB per chip, which is bascially nothing. If we did much more aggressive checkpointing, e.g. 12 checkpoints per layer, we'd still only be at 8GB per chip. We're nowhere near being memory bound during training at these scales.

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaways**: It is technically possible to train even very large models on very small topologies, with the caveat that they will likely take a long time. Being able to calculate the total FLOPs of a training run allows us to ballpark its training time by assuming a modest MFU and a known topology.</p>

### How to shard LLaMA 3-70B for training

Let's stick to our setting from above and say we want to train LLaMA 3-70B with 4M token batch size (1024 sequences of length 4096 per batch) on a TPU v5p pod of 8960 chips. Let's discuss what the best sharding strategy is for this model.

**Question:** Under the assumptions above, can we train our model with FSDP alone? To start, let's say we can't do any sequence/context parallelism. _This should be the first idea you have, since it's simple and will introduce no extra communication if it works._

{% details Click here for the answer, once you've thought about it! %}

**Answer**: This answer will be a little pedantic. As noted above, LLaMA 3-70B is initially trained with sequences of length 4K, so the batch size of 4M tokens gives us a *sequence batch size* of 1024. That means we can only really do pure data parallelism/FSDP up to 1024 chips _because that's how many sequences we have to do data parallelism over_. So the answer in the simple sense of "fully data parallelism with no extra communication" is no. The next question will answer a slightly less pedantic version of this.

{% enddetails %}

**Question:** Let's relax the requirement of not doing any sequence sharding. If we allow ourselves to do FSDP over both the batch _and_ sequence axes, can we train LLaMA 3-70B with only FSDP on 8960 chips?

{% details Click here for the answer, once you've thought about it! %}

**Answer**: Now that we're allowing ourselves to do sequence/context parallelism as well, we can scale up way more. First let's calculate our per-device batch size. If we do 8960-way FSDP, we end with a per-TPU batch size of `4 * 1024 * 1024 / 8960 = 468 tokens`. We know from the previous section that we become ICI-bound by FSDP when $$\text{per device batch size} < 2550 / M_X$$. Since we can dedicate 3 axes here with a full 3D pod, this would give us a lower bound of 850, which we're well below. **So the answer is no, even with 3 axes. We would be solidly communication-bound.**

{% enddetails %}

**Question:** Now let's look at mixed tensor parallelism and FSDP. Does there exist some combination that lets us remain compute-bound? What amount of FSDP and tensor parallelism should we do if so?

{% details Click here for the answer, once you've thought about it! %}

**Answer**: First let's check to see if this will even fit. We know that we'll be comms-bound if our per-chip batch size is less than $2550^2 / 2F = 113$. As we saw above, we're slightly above this. So that's great! Now to pick the optimal amount of FSDP, we can use the formula

$$X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618$$

2-ын ойролцоо утгад ойртуулж тоймлон бодвол, бидэнд ойролцоогоор 2048 FSDP болон 4 tensor parallelism байна. Энэ нь сайн ажиллах ёстой!

{% enddetails %}

<p markdown=1 class="takeaway">**Гол санаа**: Бид LLaMA-3 загварыг 4 сая токен бүхий batch size-тайгаар бүтэн TPU v5p pod дээр сургалт хийж чадна. Үүнд өгөгдлийн параллелизм (1024-удаа), дарааллын параллелизм (2-удаа), болон тэнцэрийн параллелизм (4-удаа) хослуулж хэрэглэнэ. Ингэснээр харилцааны (communication) асуудал үүсэхгүй. Хэрвээ зөвхөн FSDP эсвэл FSDP + дарааллын параллелизм хэрэглэвэл бид харилцааны асуудалтай болно. Өмнөх хэсэгт гаргасан томъёонууд маш хэрэгтэй, практик юм.</p>

## Ажилласан бодлогууд

**Асуулт 1 [LLaMA 70B-г олон чип дээр өргөжүүлэх]:** Бид LLaMA 3-70B-г 4 под дээр ижил batch size-тайгаар сургахыг хүсвэл ямар параллелизмын схем ашиглах вэ? Бид тооцоолол (compute) эсвэл харилцаа холбоо (communication)-нд хязгаарлагдах уу? Сургах хугацаа ойролцоогоор хэр удаан байх вэ? *Зөв roofline bound-ыг ашиглахаа мартуузай.*

**Асуулт 2 [LLaMA 405B]:**

(a) LLaMA 3-405B [config](https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json)-ийг ашиглан, дээрхтэй адил бүх гол hyperparameter-уудыг агуулсан хүснэгт бичнэ үү. Энэ загвар нийт хэдэн параметртэй вэ? Нэг сургалтын алхамд хэдэн FLOP зарцуулдаг вэ? Хэрвээ бид 15T токен дээр сургах бол нийт хэдэн FLOP зарцуулах вэ?

(б) Бид 8 TPU v5p pod дээр сургалт хийхийг хүсэж байна гэж үзье. Ямар параллелизмын схем ашиглах вэ? Сургалт хэр удаан үргэлжлэх вэ? Тооцоолол (compute) эсвэл харилцаа холбоо (comms) аль нь хязгаарлах вэ?

<h3 markdown=1 class="next-section">Энэ бол 6-р хэсгийн бүх зүйл. 7-р хэсэг буюу Transformer inference-ийн тухай бол [энд](../inference) дарна уу.</h3>