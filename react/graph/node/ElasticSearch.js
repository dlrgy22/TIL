async function run(client) {
    const { body } = await client.search({
        index : 'random_data'
    })

    data = body.hits.hits.map((value, index) => {
        return value._source;
    })
    
    return data
}

async function get_data(client) {
    const data = await run(client);
    console.log(data)
}

const { Client } = require('@elastic/elasticsearch')
const client = new Client({ node: 'http://localhost:9200', log : 'trace' })

data = get_data(client)
