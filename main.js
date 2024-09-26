/*
 * @Author: Bin
 * @Date: 2023-11-23
 * @FilePath: /macbert/main.js
 */
import http from 'http'
import querystring from 'querystring'
import url from 'url'

import { env, BertTokenizer, BertForMaskedLM } from '@xenova/transformers'

/**
 * Returns the value and index of the maximum element in an array.
 * @param {number[]|TypedArray} arr array of numbers.
 * @returns {number[]} the value and index of the maximum element, of the form: [valueOfMax, indexOfMax]
 * @throws {Error} If array is empty.
 */
function max(arr) {
    if (arr.length === 0) throw Error('Array must not be empty')
    let max = arr[0]
    let indexOfMax = 0
    for (let i = 1; i < arr.length; ++i) {
        if (arr[i] > max) {
            max = arr[i]
            indexOfMax = i
        }
    }
    return [max, indexOfMax]
}

class MyClassificationPipeline {
    static task = 'text-classification'
    static model = null
    static instance = null

    static tokenizer = null

    static async get_errors(origin, corrected) {
        const errors = []
        let i = 0
        for await (const char of corrected) {
            if (i >= origin.length) {
                break
            } else if (char != origin[i]) {
                errors.push({
                    index: i,
                    wrong: `${origin[i]}`,
                    correct: `${char}`
                })
            }
            // console.log(i, origin[i], '=>', char);
            i += 1
        }
        return errors
    }

    // 纠错
    static async correction(text) {
        const preprocessing = await this.tokenizer(text, { padding: true, return_tensors: 'pt' })
        const regression = await this.model(preprocessing)
        const value = regression.logits[0]
        const result = []
        for (let i = 0; i < value.dims[0]; i++) {
            const begin = i * value.dims[1]
            const m = max(value.data.slice(begin, begin + value.dims[1]))
            result.push(m[1])
        }
        // console.log('result', result)
        const _text = await this.tokenizer.decode(result, {
            skip_special_tokens: true
        })

        const corrected = _text.replaceAll(' ', '')
        const errors = await MyClassificationPipeline.get_errors(text, corrected)

        return [corrected, errors]
    }

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            // NOTE: Uncomment this to change the cache directory
            env.cacheDir = './models/.cache'
            env.localModelPath = './models/'
            env.allowRemoteModels = false

            this.model = await BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")
            this.tokenizer = await BertTokenizer.from_pretrained("shibing624/macbert4csc-base-chinese/onnx")
            if (!this.tokenizer || !this.model) {
                throw new Error('tokenizer load error')
            }

            this.instance = this
        }
        return this.instance
    }
}

// Define the HTTP server
const server = http.createServer()
const hostname = '127.0.0.1'
const port = 3000

// Listen for requests made to the server
server.on('request', async (req, res) => {
    // Parse the request URL
    const parsedUrl = url.parse(req.url)

    // Extract the query parameters
    const { text } = querystring.parse(parsedUrl.query)

    // Set the response headers
    res.setHeader('Content-Type', 'application/json')

    let response;
    if (parsedUrl.pathname === '/checking' && text) {
        const pipeline = await MyClassificationPipeline.getInstance()
        const [proofread, errors] = await pipeline.correction(text)
        response = {
            original: text,
            proofread,
            errors
        }
        console.info('[checking]', { ...response, errors: errors.map(i => i.index) });
        res.statusCode = 200
    } else {
        response = { 'error': 'Bad request' }
        res.statusCode = 400
    }

    // Send the JSON response
    res.end(JSON.stringify(response))
});

server.listen(port, hostname, () => {
    console.log(`Server running at http://${hostname}:${port}/`)
});

