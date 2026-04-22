# Lecture 2 - HomePod 2018 lessons: great sound is not enough

**Course:** [Product Design for Embedded Systems](../Guide.md) | **Phase 2 - Embedded Systems**

**Previous:** [Lecture 01](Lecture-01.md) | **Next:** [Lecture 03 - Later HomePod / 2023-era platform lessons](Lecture-03.md)

---

## Why study the 2018 HomePod

The first HomePod is a useful product-design lesson because it showed a very specific failure mode:

- the speaker hardware felt stronger than the assistant

That makes it a perfect case study for embedded engineers.

It proves that:

- premium acoustics alone do not make a strong AI product

---

## What HomePod got right

The early HomePod proved that people immediately notice and value:

- strong sound quality
- good far-field microphone pickup
- premium industrial design
- simple setup
- an object that feels display-worthy in a living room

This matters because it confirms a hard truth:

- audio and physical quality are not optional in a home voice product

If the object sounds weak or feels cheap, the user notices instantly.

So the first HomePod was right about one important thing:

- a smart speaker is still a speaker

---

## What HomePod got wrong

The weakness was not the hardware. The weakness was that the assistant felt limited.

The common complaints were:

- Siri felt weaker than Alexa or Google Assistant
- everyday tasks felt limited
- the product felt too dependent on one ecosystem
- shared-space privacy behavior could become awkward or unsafe

This created a bad product asymmetry:

- strong hardware
- weaker everyday usefulness

That is the exact trap a new AI speaker must avoid.

---

## The core product lesson

The HomePod 2018 lesson is simple:

- great sound can make people notice your product
- but everyday usefulness decides whether they keep it in their home

So the product cannot be:

- amazing speaker, weak assistant

It needs to be:

- strong room object, strong household tool, and strong trust object at the same time

---

## The shared-space privacy lesson

A home AI speaker lives in shared space.

That changes the product rules.

The living room is not a phone.

It is not private by default.

So the device should assume:

- voice is public by default
- secrets are not safe by default
- identification can personalize responses
- identification should not casually authorize private data exposure

This is one of the sharpest product lessons from first-wave assistants.

Embedded consequence:

- microphone mute must be physical
- mute state must be visible
- listening/responding states must be legible across the room

---

## The utility lesson

People forgive some missing features if the device is useful every day.

That means the speaker should be reliably good at:

- timers
- reminders
- weather
- household memory
- home control
- routines
- simple follow-up conversation

This is a better target than feature theater.

Do not optimize around:

- rare demo features

Optimize around:

- repeated daily usefulness

---

## The ecosystem lesson

The first HomePod also showed the danger of ecosystem prison.

If the product only feels strong inside one closed service world, its usefulness narrows quickly.

The product lesson is:

- the core experience should be useful before the ecosystem expansion layer is added

For a local-first AI speaker, that means:

- good local voice
- good local memory
- good local control behavior
- ecosystem integration as expansion, not identity

---

## Translation for an AI smart speaker

If you are designing your own AI smart speaker, keep:

- premium room presence
- strong far-field microphones
- good spoken and ambient audio
- simple setup

Avoid:

- weak assistant capability behind premium hardware
- casual exposure of private information in shared space
- a product that only works well inside one vendor ecosystem
- over-relying on audio quality as the whole product story

---

## One sentence

HomePod 2018 teaches that great sound can launch a product, but trust and everyday usefulness decide whether the product deserves to stay in the room.

---

## Lab

Answer these three questions for your own smart-speaker concept:

1. What is the one thing users should notice immediately?
2. What is the one thing users should trust immediately?
3. What everyday task set must be excellent before you add long-tail features?

If you can answer those, you are already designing a stronger product than "good hardware with assistant features attached."

---

**Previous:** [Lecture 01](Lecture-01.md) | **Next:** [Lecture 03 - Later HomePod / 2023-era platform lessons](Lecture-03.md)
