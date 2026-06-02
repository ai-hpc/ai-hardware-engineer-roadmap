# Lecture 4 - AI smart speaker design studio

**Course:** [Product Design for Embedded Systems](../Guide.md) | **Phase 2 - Embedded Systems**

**Previous:** [Lecture 03](Lecture-03.md)

---

## The goal of this lecture

The earlier lectures were about lessons.

This lecture turns those lessons into a **concrete product direction**.

The example product is:

- a premium local-first AI smart speaker for the living room

Think of it as a HomePod-class room object, but with:

- stronger local-first intelligence
- clearer trust boundaries
- less ecosystem lock-in

---

## Step 1: define the product in one sentence

Bad definition:

- smart speaker with LLM

Better definition:

- local home AI appliance for shared space

That one sentence is stronger because it already implies:

- room placement
- appliance behavior
- trust requirements
- household use, not only solo use

---

## Step 2: decide what the object should feel like

For this product, the target feeling should be:

- domestic
- calm
- trustworthy
- premium
- quietly intelligent

It should not feel like:

- a dev kit in a shell
- a surveillance camera
- a gamer gadget
- a mini PC on display

This immediately shapes **mechanical and industrial choices**.

---

## Step 3: freeze the physical identity

For a V1 AI smart speaker, these are strong default decisions:

- no camera by default
- top microphone array
- lower speaker system
- visible physical mute control
- clean rear-bottom I/O zone
- stable vertical room-object form

Why these choices matter:

- no camera by default keeps trust simple in shared space
- top microphones help far-field pickup
- lower speakers help acoustic separation and stability
- physical mute gives hardware truth to the privacy claim
- hidden I/O reduces visible clutter

This is **product design turning into hardware layout**.

---

## Step 4: separate the internal zones

One strong smart-speaker layout uses a vertical stack like this:

1. **Top interaction zone**
   - microphones
   - mute
   - volume
   - status lighting

2. **Upper quiet zone**
   - air gap and shielding around the microphones

3. **Middle compute and thermal zone**
   - SoM or main compute board
   - carrier / main board
   - heatsink
   - airflow path

4. **Lower acoustic zone**
   - woofer
   - tweeters
   - acoustic chamber

5. **Rear-bottom service zone**
   - power
   - service I/O
   - cable exit

Why this works:

- microphones stay away from fan turbulence
- speaker vibration stays away from the mic cap
- heavy parts stay low enough for stability
- the object feels appliance-like instead of exposed

---

## Step 5: make trust visible

For a home AI device, trust cannot depend **only on the phone app**.

The product needs visible truth.

That means:

- physical mute
- mute state visible across the room
- listening state clearly visible
- response / thinking state clearly visible

If users need to guess whether the device is live, the product is weak.

That is why a **physical mute switch** is not a tiny UX detail. It is a **core product architecture decision**.

---

## Step 6: choose the right audio ambition

For a V1 smart speaker, the goal should not be:

- audiophile bragging rights

The goal should be:

- clearly premium spoken intelligibility
- believable living-room audio
- strong far-field interaction

A sensible product direction is:

- `1 woofer`
- `2 angled tweeters`
- top mic array
- active or computational tuning mindset

This gives a better product story than:

- cheapest possible mono speaker

But it also avoids pretending to be:

- a hi-fi tower

The product is a home AI appliance first, not an audiophile showcase.

---

## Step 7: decide the intelligence boundary

A good V1 product should not feel like a **shell around someone else's brain**.

For a local-first AI speaker, a strong split is:

### The device should own

- wake word
- conversation policy
- memory policy
- response style
- privacy controls
- local intelligence orchestration

### The smart-home platform should own

- device graph
- automations
- integrations
- scenes
- long-tail compatibility

That means a platform like Home Assistant should increase power, but not define product identity.

The user should feel:

- one coherent appliance

not:

- a voice skin over another system

---

## Step 8: define the V1 non-goals

Strong products are often defined as much by **what they refuse** as by what they include.

For this AI smart-speaker V1, good non-goals are:

- no camera by default
- no screen-first identity
- no dependence on another nearby device for core intelligence
- no requirement to understand raw smart-home entity names
- no ecosystem prison

This protects the product from becoming blurry.

---

## A good V1 product brief

If you compress the whole lecture into one brief, it becomes:

**Product role**
- local-first home AI appliance for the living room

**Primary strengths**
- trust
- voice interaction
- room presence
- household usefulness

**Physical rules**
- top mic array
- lower speaker system
- visible mute
- calm object
- no camera by default

**Software rules**
- local-first core behavior
- Home Assistant-compatible but not dependent
- useful before ecosystem expansion

**Product risk to avoid**
- premium hardware with weak everyday usefulness

---

## Final takeaway

The right embedded-product question is not:

- can we build this?

The better question is:

- if we build this, what kind of object will it become in the user's home?

That is the real bridge between **embedded engineering and product design**.

---

## Lab

Write a one-page V1 design brief for your own AI smart-speaker concept.

It must include:

1. one-sentence product role
2. target room or placement
3. trust model
4. physical control list
5. internal zone layout
6. local vs external software boundary
7. three V1 non-goals

If you can write that clearly, you are no longer just describing a device. You are describing a product.

---

**Previous:** [Lecture 03](Lecture-03.md)
