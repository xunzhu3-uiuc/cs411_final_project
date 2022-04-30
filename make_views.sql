USE AcademicWorld;

CREATE OR REPLACE VIEW top_keywords_by_num_citations                 AS
SELECT   RANK() OVER (ORDER BY total_num_citations DESC) AS rank,
         `name`,
         total_num_citations
FROM     (
                   SELECT    p.id,
                             SUM(p.num_citations) AS total_num_citations,
                             k.`name`
                   FROM      publication p
                   LEFT JOIN publication_keyword pk
                   ON        p.id = pk.publication_id
                   LEFT JOIN keyword k
                   ON        k.id = pk.keyword_id
                   GROUP BY  k.`name`
                   HAVING    k.`name` IS NOT NULL) AS s
ORDER BY total_num_citations DESC;

CREATE OR REPLACE VIEW top_keywords_by_num_publications AS
SELECT   RANK() OVER (ORDER BY total_num_publications DESC) AS rank,
         `name`,
         total_num_publications
FROM     (
                   SELECT    p.id,
                             COUNT(DISTINCT p.id) AS total_num_publications,
                             k.`name`
                   FROM      publication p
                   LEFT JOIN publication_keyword pk
                   ON        p.id = pk.publication_id
                   LEFT JOIN keyword k
                   ON        k.id = pk.keyword_id
                   GROUP BY  k.`name`
                   HAVING    k.`name` IS NOT NULL) AS s
ORDER BY total_num_publications DESC;
