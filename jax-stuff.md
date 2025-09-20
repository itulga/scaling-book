—
Энэхүү орчуулга нь MIT лицензийн дагуу эх бүтээлээс хөрвүүлэв.
Эх сурвалж: Austin et al., "How to Scale Your Model" (https://jax-ml.github.io/scaling-book/)
Орч.: Mongolian (mn)
—

---
layout: distill
title: "JAX дээр TPU програмчлах"
# permalink: /main/
description: "JAX ашиглан TPU-г үр дүнтэй програмчлах! Энэ хэсгийн ихэнхийг <a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>эндээс</a> авсан болно. Энэ хэсгийн кодын жишээг үнэгүй TPU-тайгаар <a href='https://colab.sandbox.google.com/'>Google Colab</a>-д ажиллуулж болно."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

хэсгийн_дугаар: 10

previous_section_url: "../profiling"
previous_section_name: "9-р хэсэг: Профайлинг"

next_section_url: ../conclusion
next_section_name: "11-р хэсэг: Дүгнэлтүүд"

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
  - name: Яш Катариа
    url: https://x.com/yashk2810
  - name: Рейнер Поп<sup>*</sup>
    url: https://x.com/reinerpope

# Өөрийн бичлэгт агуулгын жагсаалт нэмэх.
#   - TOC (Агуулгын жагсаалт)-ийн нэрүүд нь тухайн хэсгийн нэртэй яг ижил байх ёстой,
#     ингэснээр бичлэг доторх холбоосууд зөв ажиллана.
#   - Доорх форматыг ашиглана уу, markdown агуулгын жагсаалтыг гараар хийхээс зайлсхий.
toc:
  - name: "JAX-д Параллелизм хэрхэн ажилладаг вэ?"
  - subsections:
    - name: "Автомат хуваах горим"
    - name: "Ил тод хуваах горим"
    - name: "shard_map ашиглан гараар хуваах горим"
  - name: "Бодлогын жишээнүүд"

# Доорх нь нэмэлт постод зориулсан тусгай загвар (styles) оруулах жишээ юм.
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

## JAX-д Параллелизм хэрхэн ажилладаг вэ?

JAX нь олон төхөөрөмж дээр програм бичих гурван төрлийн арга барилыг дэмждэг:

1. **Компилер, жолоогоо барь!** XLA компилер автоматаар массивуудыг хувааж, тухайн програмыг ажиллуулахад шаардлагатай харилцааг нэмэхийг шийддэг. Энэ нь нэг төхөөрөмж дээр ажилладаг програмыг ямар ч өөрчлөлтгүйгээр мянга мянган төхөөрөмж дээр автоматаар ажиллуулах боломж олгодог.

2. **JAX, жолоогоо барь!** Автомат параллелизм бол сайн, гэхдээ заримдаа компилер хачин зүйл хийдэг. Ил тод sharding ашигласнаар та нэг төхөөрөмжийн кодоо ердийнхөөрөө бичиж болно, харин JAX нь sharding тархалтыг (компилер биш) зохицуулна. Энэ нь JAX-д таны хүсэл тодорхойгүй үед асуух боломж олгодог.

3. **Зүгээр л би юу бичмээр байгаагаа бичье, ээ бурхан минь!** Компилерүүд сайхан ч, заримдаа буруу зүйл хийж, таны хүсээгүй харилцааг нэмдэг. Заримдаа бид яг ямар харилцаа хийхийг хүсэж байгаагаа тодорхой бичихийг хүсдэг.

| Горим | Харах уу? | Ил тод хуваах уу? | Ил тод хамтын ажиллагаа уу? |
|:---:|:---:|:---:|:---:|
| Автомат | Глобал | ❌ | ❌ |
| Ил тод | Глобал | ✅ | ❌ |
| Гараар | Төхөөрөмж бүрээр | ✅ | ✅ |

Үүнтэй адил, JAX нь эдгээр горим бүрт зориулсан API-уудыг өгдөг:

1. `jax.jit` ( `Auto` торны тэнхлэгүүдтэй ) нь танд байгаа JAX функцыг ямар ч байдлаар авч, хуваарилсан оролтуудтай дуудахад тусална. JAX дараа нь XLA-ийн [Shardy](https://openxla.org/shardy) компилерийг ашиглаж, програмыг автоматаар зэрэгцүүлнэ. XLA нь шаардлагатай үед (AllGathers, ReduceScatters, AllReduces гэх мэт) харилцааг автоматаар нэмнэ, ингэснээр одоо байгаа үйлдлүүдийг хийхэд тусална. Энэ нь төгс биш ч ихэвчлэн таны програмыг ямар ч чипийн тоонд автоматаар өргөжүүлэхэд хангалттай сайн ажилладаг.

2. `jax.jit` нь `Explicit` торны тэнхлэгүүдтэй, (1)-тэй төстэй харагддаг, гэхдээ XLA-ийн оронд JAX өөрөө хуваарилалтыг удирддаг. Энэ нь массивын хуваарилалт нь JAX-ийн төрөл системийн нэг хэсэг гэсэн үг бөгөөд JAX тодорхойгүй харилцаа илрүүлбэл алдаа гаргаж, хэрэглэгчид шийдэх боломж олгоно.

3. `jax.shard_map` нь илүү гар аргаар хийх арга юм. Та програмын төхөөрөмжид хамаарах хэсгийг харах бөгөөд хүссэн харилцаагаа өөрөө бичих хэрэгтэй болно. Хуваарилсан массивтай, бүх төхөөрөмж дээр бүрэн массив хэрэгтэй байна уу? `jax.lax.all_gather` нэмнэ үү. Массивыг бүх төхөөрөмжүүд дээр нийлүүлж нэмэхийг хүсэж байна уу? `jax.lax.psum` (AllReduce) нэмнэ үү. Програмчлах нь илүү хэцүү боловч таны хүсээгүй зүйлийг хийх магадлал бага.

<h3 id="auto-sharding-mode">Автомат хуваах горим</h3>

jax.jit нь JAX дотор хоёр үүрэгтэй. Нэрнээс нь харахад, энэ нь Python функцыг "just-in-time" буюу яг одоо байхад нь байт код руу (XLA/HLO/LLO ашиглан) хөрвүүлдэг тул илүү хурдан ажилладаг. Гэхдээ хэрвээ оролт нь sharded буюу хуваагдсан эсвэл хэрэглэгч `in_sharding` эсвэл `out_sharding`-г заасан бол XLA нь тооцооллыг олон төхөөрөмж дээр хуваарилж, шаардлагатай бол харилцаа холбоо нэмэх боломжтой болдог. Жишээ нь, энд jax.jit ашиглан хэрхэн sharded matmul бичихийг харуулж байна:

```py
import jax
import jax.numpy as jnp

# Running on an TPU v5e 4x2. This assigns names to the two physical axes of the hardware.
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# This tells JAX to use this mesh for all operations, so you can just specify the PartitionSpec P.
jax.set_mesh(mesh)

# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# We can explicitly compile the sharded matmul function here. This adds all the
# necessary comms (e.g. an AllReduce after the matmul).
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

Энэ нь ямар ч sharding ашигласан байсан автоматаар ажиллана, мөн тооцооллыг манай төхөөрөмжүүд дээр хуваана. **Гэхдээ hardware түвшинд яг юу болж байна вэ?**

1. Эхлээд бид In болон W массивуудыг төхөөрөмжүүд дээрээ хуваана<d-footnote>Энд бид хэрхэн хийснийг анхаарна уу. Энэ бол массивыг тодорхой хуваалттайгаар үүсгэх нэг арга юм (өөрөөр хэлбэл, үүсгэх функцэд device аргумент нэмэх). Өөр нэг арга нь массивыг энгийнээр `jnp.array(....)` ашиглан үүсгээд дараа нь жишээ нь `jax.device_put(..., P('x', 'y'))` хийх юм. Мөн хүссэн массивыг үүсгэдэг функц бичээд, үүнийг jit-compile хийхэд `out_shardings`-г хүссэнээрээ ашиглаж болно.</d-footnote>. W массивыг contracting хэмжээний дагуу 2 хувааж, харин In массивыг contracting болон output хэмжээний дагуу 4 хуваана. Энэ нь W[D<sub>Y</sub>, F] болон In[B<sub>X</sub>, D<sub>Y</sub>] гэсэн хуваалттай, өөрөөр хэлбэл model болон data parallelism гэсэн хоёр төрлийн параллелчлалтай тэнцэнэ.

2. Хэрвээ бид үүнийг локал (өөрөөр хэлбэл нэг төхөөрөмж дээр) ажиллуулж байсан бол `matmul_square` нь оролтыг квадратад оруулаад энгийн матриц үржүүлэлт хийх байсан. Гэвч бид `out_shardings`-г `P('X', None)` гэж заасан тул гаралт нь batch хэмжээний дагуу хуваагдаж, model хэмжээний дагуу давхардах бөгөөд AllReduce хэрэгтэй болно.

Өмнөх хэсгүүдийн тэмдэглэгээг ашиглавал, энэ нь магадгүй дараах шиг зүйл хийх байх

1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F] { U<sub>Y</sub> })

`jax.jit` үүнийг бидний оронд автоматаар нэмнэ! Бид үнэхээр `jit_matmul.as_text()` ашиглан HLO-г хэвлэж, дараах HLO-г (маш товчлон) харж болно:

```py
# This fusion is the actual matmul of the sharded inputs and matrix
%fusion = bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[2,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# We reduce the partially summed results across devices
ROOT %AllReduce = bf16[2,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

Бид дээр матmul (нэгдэл) болон AllReduce-ийг харж байна. Хэлбэрүүдэд онцгой анхаарал хандуулаарай. `bf16[2, 1024]` нь идэвхжүүлэлтийн local харагдац юм, учир нь бидний `batch_size=8` нь 4 төхөөрөмж дээр хуваагдсан бөгөөд бидний `d_model=2048` мөн адил 2 янзаар хуваагдсан байна.

**Энэ үнэхээр ид шидтэй!** Манай програм ямар ч төвөгтэй байсан ч, [Shardy](https://openxla.org/shardy) болон jit нь бүх завсрын идэвхжүүлэлтийн (intermediate activations) хувьд shard хийх аргуудыг хайж, шаардлагатай бол харилцаа холбоо (communication) нэмэхийг оролддог. Гэхдээ, Shardy-д дутагдал бий. Заримдаа алдаа гаргаж болдог. Заримдаа та профайл (profile)-ыг хараад ямар нэг зүйл буруу болсон байгааг анзаарна. Гигант AllGather нь профайлын 80%-ийг эзэлдэг, тэгэх шаардлагагүй байхад. Ийм зүйл тохиолдвол бид компайлэрийг (compiler) засах гэж оролдож, завсрын тэнцэтгэлүүдийг (intermediate tensors) илүү тодорхой аннотаци (annotation) хийж `jax.lax.with_sharding_constraint`-ээр зааж өгч болно. Жишээ нь, хоёр matmul ашиглаж байгаа үед би завсрын идэвхжүүлэлтийг `y` хэмжээст дагуу shard хийж болно (энэ нь сайн санаа биш байж магадгүй ч), дараах байдлаар:

```py
import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X', 'Y'))

def matmul(x, Win, Wout):
  hidden = jnp.einsum('bd,df->bf', x, Win)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('x', 'y'))
  return jnp.einsum('bf,df->bd', hidden, Wout)
```

Энэ нь автоматаар хуваах ертөнцөд JAX-ийн зэрэгцээ програмчлалын ойролцоогоор 60%-ийг эзэлдэг бөгөөд та завсрын shard-уудыг `jax.lax.with_sharding_constraint`-ээр удирддаг. Гэхдээ "compiler tickling" гэдэг нь алдартайгаар хөгжилтэй програмчлалын загвар биш. Та бүх завсрын хувьсагчийг тэмдэглэж болох ч зөв үр дүнд хүрэх эсэхээ мэдэхгүй байж болно. Үүний оронд, хэрвээ JAX өөрөө shard-уудын тархалтыг удирдаж, хянаж чадвал яах вэ?

<h3 id="explicit-sharding-mode">Ил тод хуваарилалтын горим</h3>

Ил тод шардлах (эсвэл “төрлөөр шардлах”) нь автоматаар шардлахтай их төстэй харагддаг, гэхдээ шардлах дамжуулалт нь JAX түвшинд явагддаг! JAX-ийн үйлдэл бүр нь шардлах дүрэмтэй бөгөөд энэ нь тухайн үйлдлийн аргументуудын шардлалтыг авч, үйлдлийн үр дүнгийн шардлалтыг үүсгэдэг. Та үүссэн шардлалтыг `jax.typeof` ашиглан харж болно:

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

# Running on an TPU v5e 2x2. This assigns names to the two physical axes of the hardware.
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'),
                                       axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))

# This tells JAX to use this mesh for all operations, so you can just specify the PartitionSpec P.
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16).reshape(8, 2), P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # bfloat16[8@X,2@Y]
  out = x * 2
  print(jax.typeof(out))  # bfloat16[8@X,2@Y]
  return out

f(x)
```

Та харж байгаачлан, JAX нь оролтын (`x`) шардингийг гаралт руу (`x`) дамжуулсан бөгөөд эдгээрийг trace-time дээр `jax.typeof` ашиглан шалгаж болно. Ихэнх үйлдлүүдэд эдгээр дүрэм нь энгийн бөгөөд ойлгомжтой байдаг, учир нь зөвхөн нэг боломжит сонголт байдаг (жишээ нь, elementwise үйлдлүүд ижил шардингийг хадгална). Гэхдээ зарим үйлдлүүдэд үр дүнг хэрхэн shard хийх нь тодорхойгүй байдаг тул JAX нь trace-time дээр алдаа гаргаж, программистаас `out_sharding` аргументаа тодорхой өгөхийг хүсдэг (жишээ нь, jnp.einsum, jnp.reshape гэх мэт). Одоо зөрчилдөөн гардаг өөр нэг жишээг үзье:

```py
# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # This will error
```

Энэ код `Contracting dimensions are sharded and it is ambiguous how the output should be sharded. Please specify the output sharding via the `out_sharding` parameter. Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)` алдаа гаргаж байна

Энэ нь гайхалтай байна, учир нь einsum-ийн гаралтыг хэрхэн хуваах нь тодорхой биш байдаг. Гаралтын хуваалт нь дараах байж болно:
* P('X', 'Y') энэ нь reduce-scatter үүсгэнэ эсвэл
* P('X', None) энэ нь all-reduce үүсгэнэ

Auto горимтой харьцуулахад, explicit горим нь ойлгомжгүй харилцаа илрэхэд алдаа заадаг бөгөөд хэрэглэгчдээс үүнийг шийдэхийг шаарддаг. Тиймээс энд та дараахыг хийж болно:

```py
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

Auto горим ба Explicit горимыг `jax.sharding.auto_axes` болон `jax.sharding.explicit_axes` API-уудаар хослуулж болно. Илүү их мэдээлэл авахыг хүсвэл энэ [сайхан бичиг баримтыг уншаарай](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html).

<h3 id="manual-sharding-mode-via-shard_map">shard_map: програмыг ил тод хуваах, зэрэгцээ ажиллагааг удирдах</h3>

Shardy бол "compiler жолоодох" горим бол jax [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) нь бүх зүйлийг таны гарт өгдөг. Та оролтын sharding-ийг jax.jit шиг заана, гэхдээ дараа нь бүх харилцааг (communication) шууд бичнэ. `jax.jit` нь танд програмын бүх төхөөрөмжийг хамарсан (global cross-device) харагдац өгдөг бол, `shard_map` нь зөвхөн тухайн төхөөрөмж дээрх (local per-device) харагдац өгдөг.

Энд нэг жишээ байна. Энэ функц юу хийж байгааг ойлгож оролдоорой:<d-footnote>Хэрвээ та өөрөө colab дээр mesh дуурайлган туршиж үзэхийг хүсвэл, дараах cell-ийг ашиглаж болно `import jax; jax.config.update('jax_num_cpu_devices', 8)`</d-footnote>

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

mesh = jax.make_mesh((2, 4), ('x', 'y'), (shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=P(('x', 'y')))

# This function will operate on 1/8th of the array.
@jax.shard_map(in_specs=P(('x', 'y')), out_specs=P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
assert out.shape == (4,)
```

**Энэ юу хийдэг вэ?** `slice_and_average` нь тус бүр 1/8 массивтай TPU бүр дээр ажиллана, тэндээс бид эхний 4 элементийг тасдаж аван, бүх mesh дээр дундажлана. Энэ нь бид үнэндээ `mean(x[:4], x[64:68], x[128:132], …)` хийж байна гэсэн үг. Энэ их сонирхолтой, учир нь ийм үйлдлийг JAX дээр өөрөөр илэрхийлэх амаргүй.

**Яагаад үүнийг jax.jit-ийн оронд хийх вэ?** Хэрвээ бид `jax.jit`-г ашигласан бол, `slice_and_average` нь массивын бүхэл бүтэн (бүрэн `[512,]` массив) харагдах байсан. Бид энэ жигд бус хэсгийг тасдаж аваад, дараа нь дунджаар бодох хэрэгтэй болох байсан ба XLA үүнийг зөв ойлгох ёстой болно. XLA буруу холбоо нэмэх эсвэл будилах магадлалтай. Энд бид локал харагдацтай байна, зөвхөн хэрэгтэй холбоог бичиж байна.

**Жишээ [Collective Matmul]:** Илүү бодит жишээ авахын тулд, бид model parallelism хэрэгжүүлэх гэж байгаа гэж бодъё. Энд activations анхнаасаа model sharded байгаа, өөрөөр хэлбэл A[B<sub>X</sub>, D<sub>Y</sub>] \* W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]. Энгийнээр бодвол, бид эхлээд A-г AllGather хийж, дараа нь локал матриц үржүүлэлт хийх байсан:

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>])
2. Out[B<sub>X</sub>, F<sub>Y</sub>] = A[B<sub>X</sub>, D] *<sub>D</sub> W[D, F<sub>Y</sub>]

Харамсалтай нь, энэ нь муу. Учир нь бидэнд харилцаа болон тооцооллыг давхардуулах боломж олгодоггүй. Эдгээрийг давхардуулахыг "collective matmul" ашиглан хийж болно. Үүнийг [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959) дээр тайлбарласан. Энэ алгоритм үндсэндээ дараах байдлаар байна:

* Y бүрийн shard-д, A-ийн локал хэсгийг W-ийн локал хэсэгтэй matmul хийж, `[B / X, F / Y]` хэлбэртэй үр дүн гаргана. Үүний зэрэгцээ, A-г сольж дараагийн хэсгийг локалд авна, matmul хийж, үр дүнг нийлүүлж нэмнэ.

Бид үүнийг shard_map ашиглан маш амархан хэрэгжүүлж чадна:

```py
import functools

import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np

mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=('X', 'Y'),
                                       axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, jax.P('X', 'Y'))
W = jax.device_put(W, jax.P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=jax.P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs is the looped operand; rhs is the local operand
  axis_size = jax.lax.axis_size('Y')  # axis_size = 4 for this example
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # Matmul for a chunk
    update = lhs @ rhs_chunk
    # Circular shift to the left
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pvary(accum, ('X', 'Y'))
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # Compute the last chunk after the final permute to leave lhs in the state we found it
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(jax.shard_map(
  collective_matmul_allgather_lhs_contracting,
  in_specs=(jax.P('X', 'Y'), jax.P(None, 'Y')), out_specs=jax.P('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

Энэ их дажгүй байна! Бид үүнийг benchmark хийж, мөн энэ нь илүү хурдан байгааг харж болно! [Энд](https://imgur.com/a/e9I6SrM) default jit matmul-ийн profile байна. Энэ нь эхэндээ том blocking AllGather хийж, 311 микросекунд зарцуулж байна:

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}

Мөн [энд](https://imgur.com/a/21iy0Sv) дээр байгаа хувилбар нь 244 микросекунд зарцуулдаг. Та профайлд AllGather байхгүй байгааг харж болно. Энэ нь бүгд хэрэгтэй ажил юм! Манай FLOPs ашиглалт бас их өндөр байна.

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}

Мөн анхаарах хэрэгтэй зүйл нь contracting dimension дээр sharding хийгээгүй үед matmul-ийн хугацаа [224us](https://imgur.com/a/i3gNKfq) байна, тиймээс бид энд unsharded baseline-д маш ойрхон байна. Энэ нь TPU ашиглалтыг сайжруулахын тулд хийх боломжтой performance engineering-ийн нэг сайн жишээ юм. Илүү олон `shard_map` жишээ үзэхийг хүсвэл [энэ тэмдэглэл](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side) маш сайн.

Одоо энд `jax.jit` эсвэл `shard_map` ашиглан хэрэгжүүлж үзэхэд тохиромжтой хэдэн ашигтай бодлого байна!

## Ажилласан бодлогууд

Энд зарим санамсаргүй JAX-тэй холбоотой бодлогууд байна. Би дараа нь өөр бодлогууд нэмнэ. Эдгээр бүх бодлогод, танд Colab-д хэдэн TPU хэрэгтэй болно. Та олон нийтийн Colab дээр TPUv2-8 ашиглаж болно. Одоо цаашид бид танд N төхөөрөмж байгаа гэж үзнэ.

**Асуудал 1:** **A** нь float32[S<sub>X</sub>, D<sub>Y</sub>] хэлбэртэй идэвхжүүлэлтийн массив байна, `X * Y = N`-тэй. Дараахыг хий:

1. JAX дээр функц бичээрэй. Энэ функц нь `(X, Y)` shard бүрийн дундажийг тооцоолно, өөрөөр хэлбэл [X, Y] хэмжээтэй массив буцаана. Энд `arr[i, j]` нь shard `(i, j)`-ийн дундаж байна. Үүнийг `jax.jit` болон `shard_map` хоёулангаар нь хийгээрэй. Тус бүрийг profile хийгээд хэр удаан хугацаа зарцуулсныг хар. Ямар нэгэн communication нэмэгдсэн үү? *Санамж: байх ёсгүй, гэхдээ заримдаа XLA үүнийг нэмдэг.*

2. JAX дээр дараах функцийг бичнэ үү. Энэ функц нь roll(x, shift, axis=0) - x-г буцаана. shift нь **шард бүрийн X дотор** байна. Би та нарыг jax.jit ашиглаж зовооё гэж бодоогүй, тиймээс зүгээр л `shard_map` ашиглаарай.

{% details Хариуг харахын тулд энд дарна уу. %}

1-р хэсэг: Энэ бол 1-р хэсгийн шийдэл юм. `jax.jit` шийдэлд бид нэлээд төвөгтэй reshape хийх хэрэгтэйг анхаарна уу.

```py
import numpy as np

import jax
import jax.numpy as jnp

P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((4, 2), ('X','Y'))

average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=P('X','Y'), out_specs=P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

2-р хэсэг: Энэ бол 2-р хэсгийн төстэй шийдэл юм.

```py
import numpy as np

import jax
import jax.numpy as jnp

import functools

P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((4, 2), ('X','Y'))

def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0),
      mesh=mesh,
      in_specs=P('X','Y'), out_specs=P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return jnp.roll(reshaped, shift, axis=1).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

{% enddetails %}

**Асуудал 2:** Энд бид хамтдаа энгийн "экспертүүдийн холимог" загвар хийх болно. **W**: float32[E<sub>X</sub>, D, F<sub>Y</sub>] нь E ширхэг "эксперт" матрицын багц байна. **A**: float32[S<sub>X</sub>, D<sub>Y</sub>] (бидний идэвхжүүлэлтүүд) байна. Мөн **B** нь "маршрутын оноолт"-ын багц бөгөөд B[i] нь `[0, E)` мужид байх бүхэл тоо бөгөөд аль матрицаар тухайн идэвхжүүлэлтийг боловсруулахыг заана. JAX дээр бид `Out[i] = W[B[i]] @ A[i]`-г буцаадаг функц бичихийг хүсэж байна.

1. Эхлээд шардинг-ийг бүрэн орхиж үзье. Эдгээр tensor-уудыг бүгдийг нь нэг төхөөрөмжид багтахаар жижиг болго. Энэ функцийн локал хэрэгжүүлэлтийг бич. *`[S, D, F]` хэлбэртэй array-г бүтээж болохгүй гэдгийг анхаараарай! Санаа: token-уудыг `[E, S, D]` хэлбэртэй шинэ buffer руу эрэмбэлж хийхийг оролдоорой, маск хийхэд анхаараарай (хоёр дахь хэмжээс яагаад S хэмжээтэй байх хэрэгтэй вэ?).*

2. Хэрвээ та дээрх аргыг зүгээр л `jax.jit` бол ямар нэг зүйл болно. Үүнийг profile хийж, ямар харилцаа хийхийг нь хараарай. Энэ хэр удаан хугацаа зарцуулж байна вэ?

3. Дээрх асуудлын нэг нь бүх идэвхжүүлэлтийн **A**-г локал дээр цуглуулдаг явдал юм, жишээ нь AllGather<sub>X</sub>([S<sub>X</sub>, D<sub>Y</sub>]). Энэ нь зөвхөн харилцаа холбооны хувьд их зардалтай биш, бас бүх идэвхжүүлэлтийг локал дээр багтааж чадахгүй бол санах ойн хувьд маш их зардалтай. Дээрхийг `shard_map` болон ил тод харилцаа холбоо ашиглан хэрэгжүүлээрэй.

      1. Эхний удаад, `jax.lax.all_gather` ашиглах нь хамгийн амархан байж болох бөгөөд (a)-д байгаа шиг дахин дараалалд оруулж болно.

2. Хоёр дахь удаагийн туршилтаар, `[E, S, D]` хэмжээтэй ямар ч array үүсгэхээс зайлсхийхийг хичээ. Өөрөөр хэлбэл, тооцооллыг ragged байдлаар, `jax.lax.while_loop` дотор `jax.lax.all_to_all` ашиглан хийж үзээрэй. Ингэснээр, бүх activations-ийг бүтнээр нь үүсгэхгүй бөгөөд padding дээр илүү compute зарцуулахгүй байх боломжтой. Энэ нь таны анхны хэрэгжүүлэлтээс хэр хурдан байна вэ?

4. Ихэнх MoE системүүд олон (k) эксперт рүү чиглүүлж, дараа нь үр дүнг дундажладаг. Дээрх кодыг үүнийг хэрэгжүүлэхээр өөрчил. Энэ тохиолдолд **B**: int32[S, k] байна, k эксперт рүү чиглүүлэхэд ашиглагдана.

**Асуудал 3:** Дээрх хамтын matmul жишээ нь үнэхээр бодит LLM-д маш их хамаатай. Одоо энэ жишээг өөрчилж, бүтэн Transformer stack хийх гэж оролдоё.

1. Дасгал болгон, AllReduce collective matmul хэрэгжүүлье, өөрөөр хэлбэл A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]. Анхаар: гаралт нь давхардаагүй байна. Энгийн алгоритмыг дээр тайлбарласан, үндсэндээ зөвхөн локал matmul хийж, дараа нь AllReduce ашиглана. Энэ үйлдлийн comms давхцсан "collective" хувилбарыг хийхийг оролдоорой. *Санамж: гаралтын хэмжээсээр tile хийж болно, мөн `jax.lax.psum` (өөрөөр хэлбэл AllReduce) ашиглахад чөлөөтэй байгаарай.* *Тэмдэглэл: XLA үүнийг хэрхэн зохицуулдгаас шалтгаалаад, энэ нь суурь хувилбараас хурдан байхгүй байж магадгүй.*

2. Дээрх AllReduce хамтын матмулын эсрэг нь ReduceScatter хамтын матмул юм. Жишээ нь: Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]. Энэ нь Transformer-ийн доош чиглэсэн матриц дээр тохиолддог. Үүний хамтын, давхцсан хувилбарыг JAX дээр хэрэгжүүлээрэй. Зөвхөн танд хэрэгтэй хамгийн бага өгөгдлийг дамжуулж байгаарай. *Санамж: үр дүнг хуримтлуулж байхдаа байрлалыг нь сольж үзээрэй.*

3. Эдгээр хоёрыг нийлүүлээд, эхнээс нь дуустал Transformer блок болгоорой. Энэ блок нь In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>] гэсэн үйлдлийг давхцсан (overlapped) холбоотойгоор гүйцэтгэнэ.<d-footnote>Өмнөхтэй адил, бид энд нэг non-linearity (шугаман бус функц)-г орхисон тул $W_{in} \cdot W_{out}$-г эхэлж хийж болохгүй.</d-footnote> Энэ нь `jax.jit` хэрэгжүүлэлтээс хэр хурдан бэ?

**Асуудал 4:** Дээрх бүх collective matmul-ууд нэг чиглэлтэй: тэд зөвхөн нэг чиглэлд permute хийдэг. Collective AllReduce matmul болон collective ReduceScatter matmul-уудыг хоёр чиглэлтэй харилцаа ашиглахаар дахин бичнэ үү. Эдгээр нь хэр хурдан болох вэ?

### 10-р хэсэг дууслаа. Үндсэндээ энэ бол бүх зүйл! Эцсийн дүгнэлт болон цааш унших бол [энд](../conclusion) дарна уу.