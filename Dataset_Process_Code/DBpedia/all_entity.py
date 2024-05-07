from SPARQLWrapper import SPARQLWrapper,JSON
sparql = SPARQLWrapper("http://localhost:8890/sparql/")


for i in range(0, 1000):
    
    offset = 1048576 * i
    sparql.setQuery("""
    PREFIX dbo: <http://dbpedia.org/ontology>
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT 
    ?entity
    FROM <https://dbpedia.org>
    WHERE
    {
        {?entity ?p ?o.
        FILTER strstarts(str(?entity), 'http://dbpedia.org/resource/')}
    }
    LIMIT 1048576
    OFFSET """ + str(offset) +"""
    """
    )


    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    print(len(results['results']['bindings']))

# entity_list=[]
# for i in range(len(results['results']['bindings'])):
#     if results['results']['bindings'][i].get('entity') is not None:
#         entity_list.append(results['results']['bindings'][i]['entity']['value'].split('/')[-1])
#     else: entity_list.append(None)

    

    with open('/data/c_x/dbpedia_entity/all_ent_{}.txt'.format(i),'w+') as f:
        for i in results['results']['bindings']:
            f.write(i['entity']['value'].split('/')[-1] + '\n')
    
    if len(results['results']['bindings']) < 1048576:
        break


