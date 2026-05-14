import asyncio
import edge_tts

texto = "Hola, bienvenido a PayAI. El sistema está funcionando correctamente."

async def gerar_audio():
    communicate = edge_tts.Communicate(
        text=texto,
        voice="es-CO-GonzaloNeural"
    )

    await communicate.save("voz.mp3")

asyncio.run(gerar_audio())

print("Áudio gerado!")