---
title: Sentinel Chat
emoji: 🛡️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Multimodal content moderation across 12 safety categories.
---

# Sentinel Chat — live demo

A WhatsApp-style chat where every message and attachment is screened before it
lands in the conversation: curse words and PII are masked, explicit images are
blurred, dangerous files and serious harms are blocked.

This Space runs the **Hugging Face text + image models** (toxic-bert + KoalaAI
moderation + NSFW ViT) on top of the offline floor (lexicon, PII regexes,
file/EICAR gate, NudeNet) — so it catches implicit toxicity and hate, not just
listed words. The models are baked into the image, so the first message is
instant. Whisper audio is off by default to keep CPU latency low. Source and
full docs: https://github.com/stevenkayitaresteven/portifolio
