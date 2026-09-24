package com.example.executorchllamademo.rag

import android.content.Context
import android.net.Uri
import android.util.Log
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import android.graphics.Bitmap
import android.graphics.pdf.PdfRenderer
import android.os.ParcelFileDescriptor
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.tasks.await

/** Extracts text from clinical PDFs, with an OCR fallback, and splits it into chunks for RAG ingestion. */
class ClinicalPdfProcessor(private val context: Context) {

    init {
        PDFBoxResourceLoader.init(context)
    }

    suspend fun extractText(uri: Uri): String {
        var text = try {
            context.contentResolver.openInputStream(uri)?.use { inputStream ->
                PDDocument.load(inputStream).use { document ->
                    PDFTextStripper().getText(document)
                }
            } ?: ""
        } catch (e: Exception) {
            Log.e("ClinicalPdfProcessor", "PDFBox error: ${e.localizedMessage}")
            ""
        }

        Log.d("DEBUG_RAG", "1. Text extracted by PDFBox: [$text]")

        if (text.isBlank()) {
            Log.d("ClinicalPdfProcessor", "No text found with PDFBox. Starting OCR with ML Kit...")
            text = extractTextWithOcr(uri)
            Log.d("DEBUG_RAG", "2. Text extracted by OCR: [$text]")
        }

        return text
    }

    private suspend fun extractTextWithOcr(uri: Uri): String {
        val textBuilder = StringBuilder()
        val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

        var fileDescriptor: ParcelFileDescriptor? = null
        var pdfRenderer: PdfRenderer? = null

        try {
            fileDescriptor = context.contentResolver.openFileDescriptor(uri, "r") ?: return ""
            pdfRenderer = PdfRenderer(fileDescriptor)

            for (i in 0 until pdfRenderer.pageCount) {
                val page = pdfRenderer.openPage(i)

                val bitmap = Bitmap.createBitmap(page.width * 2, page.height * 2, Bitmap.Config.ARGB_8888)
                page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)

                val image = InputImage.fromBitmap(bitmap, 0)
                val visionText = recognizer.process(image).await()

                textBuilder.append(visionText.text).append("\n")

                page.close()
                bitmap.recycle()
            }
        } catch (e: Exception) {
            Log.e("ClinicalPdfProcessor", "Error during OCR: ${e.localizedMessage}", e)
        } finally {
            pdfRenderer?.close()
            fileDescriptor?.close()
        }

        return textBuilder.toString()
    }

    fun createClinicalChunks(
        text: String,
        maxChars: Int = 500,
        overlapChars: Int = 100
    ): List<String> {
        Log.d("DEBUG_RAG", "3. Text going into chunking (length: ${text.length}): [$text]")
        if (text.isBlank()) {
            Log.d("DEBUG_RAG", "4. WARNING: the text to be chunked is empty!")
            return emptyList()
        }

        val cleanText = text.replace(Regex("[ \\t]+"), " ").trim()
        val result = mutableListOf<String>()
        var startIndex = 0

        while (startIndex < cleanText.length) {
            var endIndex = startIndex + maxChars
            if (endIndex >= cleanText.length) {
                val lastChunk = cleanText.substring(startIndex).trim()
                if (lastChunk.isNotEmpty()) result.add(lastChunk)
                break
            }

            var cutIndex = endIndex

            val lastParagraph = cleanText.lastIndexOf("\n\n", endIndex)
            if (lastParagraph > startIndex + (maxChars / 2)) {
                cutIndex = lastParagraph + 2
            } else {
                val lastLine = cleanText.lastIndexOf("\n", endIndex)
                if (lastLine > startIndex + (maxChars / 2)) {
                    cutIndex = lastLine + 1
                } else {
                    val lastPeriod = cleanText.lastIndexOf(". ", endIndex)
                    if (lastPeriod > startIndex + (maxChars / 2)) {
                        cutIndex = lastPeriod + 2
                    } else {
                        val lastSpace = cleanText.lastIndexOf(" ", endIndex)
                        if (lastSpace > startIndex) cutIndex = lastSpace + 1
                    }
                }
            }

            if (cutIndex <= startIndex) {
                cutIndex = endIndex
            }

            val chunk = cleanText.substring(startIndex, cutIndex).trim()
            if (chunk.isNotEmpty()) result.add(chunk)

            val nextStartIndex = cutIndex - overlapChars
            if (nextStartIndex > startIndex) {
                val nextSpace = cleanText.indexOf(" ", nextStartIndex)
                if (nextSpace in nextStartIndex until cutIndex) {
                    startIndex = nextSpace + 1
                } else {
                    startIndex = nextStartIndex
                }
            } else {
                startIndex = cutIndex
            }
        }
        Log.d("DEBUG_RAG", "5. Final chunks created: ${result.size}")
        if (result.isNotEmpty()) {
            Log.d("DEBUG_RAG", "Example Chunk 1: [${result.first()}]")
        }
        return result
    }
}