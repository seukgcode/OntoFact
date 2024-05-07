from SPARQLWrapper import SPARQLWrapper,JSON
sparql = SPARQLWrapper("http://localhost:8890/sparql/")

entity_list=[]
with open('/data/c_x/dbpedia_ontology/ent.txt', 'r') as f:
    for line in f.readlines():
        entity_list.append(line.strip())

entity_description_file = open('entity_description.txt','a')
new_ontology_file = open('ontology_dbo_dbp.txt','a')
all_triple_file = open('all_triples.txt','a')
for i in range(len(entity_list)):
    if i < 920116:
        continue
    e = entity_list[i]
    sparql.setQuery(
        """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX dbr: <http://dbpedia.org/resource/>
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX dbp: <http://dbpedia.org/property/>
            select *
            from <https://dbpedia.org>
            where{
                {<http://dbpedia.org/resource/"""+ e +"""> ?p ?o.
                FILTER strstarts(str(?p), str(dbp:))}
                UNION
                {
                <http://dbpedia.org/resource/"""+ e +"""> ?p ?o.
                FILTER strstarts(str(?p), str(dbo:))}
            }
        """
    )
    sparql.setReturnFormat(JSON)
    results2 = sparql.query().convert()['results']['bindings']
    for res in results2:
        s = e
        # print(s,res)
        p = res['p']['value'].split('/')[-1]
        o_type = res['o']['type']
        o_value = res['o']['value']
        if p == 'abstract':
            entity_description_file.write(e+'\t'+o_value.replace('\n', '')+'\n')
        else:
            if o_type == 'literal':
                continue

            if o_type == 'typed-literal':
                new_o = res['o']['datatype'].split('/')[-1]
            
                new_ontology_file.write(s + '\t' + p + '\t' + new_o + '\n')

            all_triple_file.write(s + '\t' + p + '\t' + o_value.split('/')[-1] + '\n')
entity_description_file.close()
new_ontology_file.close()
all_triple_file.close()
        



