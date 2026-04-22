# Lecture 1 - Why product design belongs in embedded systems

**Course:** [Product Design for Embedded Systems](../Guide.md) | **Phase 2 - Embedded Systems**

**Next:** [Lecture 02 - HomePod 2018 lessons: great sound is not enough](Lecture-02.md)

---

## Start with the wrong mental model

Many engineers treat product design as if it begins after the real engineering work is done.

That wrong mental model sounds like this:

- hardware team builds the board
- firmware team makes it boot
- software team makes the features work
- later someone "designs the product"

That is not how good embedded products are made.

In real systems, product design shows up much earlier in questions like:

- should this object live on a desk, wall, shelf, or kitchen counter?
- should it look like equipment, furniture, or an appliance?
- what should the user trust at a glance?
- which states must be visible across the room?
- where should microphones, speakers, vents, and buttons physically live?
- what should work before any cloud or app setup exists?

Those are product questions, but they also change:

- PCB shape
- mechanical layout
- thermal path
- audio path
- EMI risk
- GPIO and button count
- software ownership boundaries

---

## The AI smart-speaker example

This course uses an **AI smart speaker / home AI appliance** as the running example because it forces many product and embedded questions into one object.

A good smart speaker is not only:

- a Linux box
- a microphone array
- a speaker driver
- an LLM endpoint

It is also:

- a room object
- a trust object
- an audio object
- a setup experience
- a household behavior system

That is why smart speakers are such a useful design exercise.

They make it obvious that product design is part of system design.

---

## Product design means system-shaping decisions

For embedded systems, product design usually means decisions in five areas.

### 1. Product role

What is this thing fundamentally trying to be?

Examples:

- dev kit
- appliance
- room assistant
- portable tool
- hidden module

This choice changes everything downstream.

A living-room AI speaker should not feel like:

- a router
- a surveillance camera
- a mini PC in a shell

### 2. Trust model

What must the user be able to understand instantly?

Examples:

- is the microphone muted?
- is the device listening?
- is it safe in shared space?
- does it contain a camera?

If trust depends only on app UI, the product is weak.

### 3. Physical interaction model

What can the user do physically?

Examples:

- mute
- volume
- pairing
- reset

A physical mute switch is not just a UI detail. It is a product claim implemented in hardware.

### 4. Environmental fit

Where does the product live?

Examples:

- shelf
- bedside table
- factory floor
- car dashboard

This changes:

- size
- orientation
- cable exit
- vent location
- acoustic tuning
- button placement

### 5. System boundary

What belongs inside the product, and what belongs outside it?

For a smart speaker, that means questions like:

- what is local-first?
- what depends on cloud?
- what is owned by the device brain?
- what is owned by Home Assistant or another ecosystem?

That is product design too.

---

## Why engineers should care

If you ignore product design, you often get a technically impressive but weak object.

Typical failure modes:

- good hardware, awkward placement
- powerful software, weak trust cues
- strong features, confusing setup
- premium parts, cheap-feeling object
- strong acoustics, weak assistant usefulness

In other words:

- the system works
- but the product loses

That is the exact gap this course is trying to fix.

---

## A simple embedded-product checklist

Before building a real embedded product, ask:

1. What room or environment is this designed for?
2. What does the object need to signal physically?
3. What user action must always work even if software is degraded?
4. What should the device own locally?
5. What external dependency would make the product feel weaker?

If you cannot answer those, the engineering is still underspecified.

---

## Running example for the rest of this course

For the rest of the lectures, use this imagined product:

- premium local-first AI smart speaker
- living-room first
- far-field microphones
- strong spoken audio
- no camera by default
- visible physical mute
- useful before ecosystem expansion

That is the lens through which the HomePod lessons and the final design studio will make sense.

---

## Lab

Write a short paragraph answering:

- what kind of embedded product do you want to design?
- where does it live?
- what must users trust instantly?
- what is one physical control that should exist no matter what?

If you can answer that clearly, you are already doing product design, not just implementation.

---

**Next:** [Lecture 02 - HomePod 2018 lessons: great sound is not enough](Lecture-02.md)
