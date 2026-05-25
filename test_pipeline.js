const axios = require('axios');
async function test() {
    try {
        const res = await axios.post('http://127.0.0.1:8000/api/research-pipeline', { query_text: 'hidden cost of owning a home' });
        console.log("Status:", res.status);
        console.log("Data:", res.data);
    } catch (e) {
        console.error("Error:", e.message);
    }
}
test();
