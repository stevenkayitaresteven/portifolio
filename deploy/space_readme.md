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

This Space runs the **toxic-bert** text model (baked into the image, served
offline) on top of the offline floor (lexicon, PII regexes, file/EICAR gate,
NudeNet) — so it catches implicit toxicity and hate, not just listed words,
while still booting fast on a free CPU box. The image ViT and Whisper audio are
off here to keep startup light (the offline NudeNet still moderates images).
Source and full docs: https://github.com/stevenkayitaresteven/portifolio
