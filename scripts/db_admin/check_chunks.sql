SELECT JSON_VALUE(metadata, '$.page_label') AS page, TEXT
FROM CIGDOCS01
WHERE JSON_VALUE(metadata, '$.source') = 'PRG_1_021-LG-CIG-2024.pdf'
order by to_number(page);